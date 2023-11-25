from agent.hyperparameters import *
import torch
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tqdm import tqdm
# from networks import ppo
from torchrl.envs import TransformedEnv, ObservationNorm, Compose, DoubleToFloat, StepCounter
from simulation.encoder_transform import EncodeImage
from autoencoder.encoder import VariationalEncoder
from torch import nn
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from collections import defaultdict
from torchrl.objectives.value import GAE


class PPOAgent:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.create_actor()
        self.create_critic()
        self.collector = SyncDataCollector(
            env,
            self.policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=device,
            storing_device=device,
            reset_at_each_iter=True,
            )
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )
        self.advantage_module = GAE(
            gamma=gamma, lmbda=lmbda, value_network=self.value_module, average_gae=True
        )

        self.loss_module = ClipPPOLoss(
            actor=self.policy_module,
            critic=self.value_module,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            # these keys match by default but we set this for completeness
            value_target_key=self.advantage_module.value_target_key,
            critic_coef=1.0,
            gamma=0.99,
            loss_critic_type="smooth_l1",
        )
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, total_frames // frames_per_batch, 0.0
        )

    def create_actor(self):
        actor_net = nn.Sequential(
            nn.LazyLinear(self.env .observation_spec["observation"].shape[-1], device=self.device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(2 * self.env .action_spec.shape[-1], device=self.device),
            NormalParamExtractor(),
        )
        actor_net.forward(torch.randn(self.env .observation_spec["observation"].shape[-1]).to(self.device))
        policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.env .action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": self.env .action_spec.space.minimum,
                "max": self.env .action_spec.space.maximum,
            },
            return_log_prob=True,
            # we'll need the log-prob for the numerator of the importance weights
        )

    def create_critic(self):
        value_net = nn.Sequential(
            nn.LazyLinear(self.env.observation_spec["observation"].shape[-1], device=self.device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(1, device=self.device),
        )
        value_net.forward(torch.randn(self.env.observation_spec["observation"].shape[-1]).to(self.device))
        self.value_module = ValueOperator(
            module=value_net,
            in_keys=["observation"],
        )
    def train(self):
        logs = defaultdict(list)
        pbar = tqdm(total=total_frames * frame_skip)
        eval_str = ""

        # We iterate over the collector until it reaches the total number of frames it was
        # designed to collect:
        for i, tensordict_data in enumerate(self.collector):
            # we now have a batch of data to work with. Let's learn something from it.
            for _ in range(num_epochs):
                # We'll need an "advantage" signal to make PPO work.
                # We re-compute it at each epoch as its value depends on the value
                # network which is updated in the inner loop.
                with torch.no_grad():
                    self.advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self.replay_buffer.extend(data_view.cpu())
                for _ in range(frames_per_batch // sub_batch_size):
                    subdata = self.replay_buffer.sample(sub_batch_size)
                    loss_vals = self.loss_module(subdata.to(self.device))
                    loss_value = (
                            loss_vals["loss_objective"]
                            + loss_vals["loss_critic"]
                            + loss_vals["loss_entropy"]
                    )

                    # Optimization: backward, grad clipping and optim step
                    loss_value.backward()
                    # this is not strictly mandatory but it's good practice to keep
                    # your gradient norm bounded
                    torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), max_grad_norm)
                    self.optim.step()
                    self.optim.zero_grad()

            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            pbar.update(tensordict_data.numel() * frame_skip)
            cum_reward_str = (
                f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
            )
            logs["step_count"].append(tensordict_data["step_count"].max().item())
            stepcount_str = f"step count (max): {logs['step_count'][-1]}"
            logs["lr"].append(self.optim.param_groups[0]["lr"])
            lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
            if i % 10 == 0:
                # We evaluate the policy once every 10 batches of data.
                # Evaluation is rather simple: execute the policy without exploration
                # (take the expected value of the action distribution) for a given
                # number of steps (1000, which is our env horizon).
                # The ``rollout`` method of the env can take a policy as argument:
                # it will then execute this policy at each step.
                with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                    # execute a rollout with the trained policy
                    eval_rollout = self.env.rollout(1000, self.policy_module)
                    logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                    logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                    eval_str = (
                        f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                        f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                        f"eval step-count: {logs['eval step_count'][-1]}"
                    )
                    del eval_rollout
            pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            self.scheduler.step()