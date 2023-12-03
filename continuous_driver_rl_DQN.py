from simulation.environment_follower_rl_DNQ import CarlaEnvironment
import carla
import time
import numpy as np
import random
# torch imports
import torch
from torchrl.envs import TransformedEnv, ObservationNorm, Compose, DoubleToFloat, StepCounter
from simulation.encoder_transform import EncodeImage
from autoencoder.encoder import VariationalEncoder
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.utils import check_env_specs
from agent.ppo_agent import PPOAgent
from torchrl.envs.transforms import CatFrames, InitTracker
from torchrl.modules import LSTMModule
import torch
import tqdm
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import (
    Compose,
    ExplorationType,
    GrayScale,
    InitTracker,
    ObservationNorm,
    Resize,
    RewardScaling,
    set_exploration_type,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ConvNet, EGreedyWrapper, LSTMModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

device = torch.device(0) if torch.cuda.device_count() else torch.device("cpu")

client = carla.Client('localhost', 2000)
world = client.get_world()
actors = world.get_actors()
for a in actors:
    if isinstance(a, carla.Vehicle) or isinstance(a, carla.Sensor):
        a.destroy()
time.sleep(0.5)
traffic_manager = client.get_trafficmanager(8000)
town = 'Town07'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
env = CarlaEnvironment(client, world, town, traffic_manager=traffic_manager, device=device)
conv_encoder = VariationalEncoder(95).to(device)

conv_encoder.load()
conv_encoder.eval()
lstm_module = LSTMModule(
        input_size=101,
        hidden_size=101,
        in_key="observation",
        out_key="observation",
        device=device,
    )
env = TransformedEnv(
    env,
    Compose(
        # normalize observations
        EncodeImage(in_keys=["pixel", 'navigation'], out_keys=["observation"], encoder=conv_encoder, del_keys=False),
        # CatFrames(5, in_keys=["observation"], out_keys=["observation"], dim=-2),
        DoubleToFloat(in_keys=["observation"]),
        StepCounter(),
        InitTracker(),
        ObservationNorm(in_keys=["observation"]),

    )
)
print(env.specs)
env.transform[-1].init_stats(num_iter=100, reduce_dim=0, cat_dim=0)
check_env_specs(env)
lstm = LSTMModule(
    input_size=101,
    hidden_size=128,
    device=device,
    in_key="observation",
    out_key="embed",
)
env.append_transform(lstm.make_tensordict_primer())
mlp = MLP(
    out_features=6,
    num_cells=[
        64, 64, 64
    ],
    device=device,
)
mlp = Mod(mlp, in_keys=["embed"], out_keys=["action_value"])
qval = QValueModule(action_space=env.action_spec, var_nums=3)
stoch_policy = Seq( lstm, mlp, qval)

stoch_policy = EGreedyWrapper(
    stoch_policy, annealing_num_steps=1_000_000, spec=env.action_spec, eps_init=0.2
)
policy = Seq( lstm.set_recurrent_mode(True), mlp, qval)
policy(env.reset())

loss_fn = DQNLoss(policy, action_space=env.action_spec, delay_value=True)

updater = SoftUpdate(loss_fn, eps=0.95)

optim = torch.optim.Adam(policy.parameters(), lr=3e-4)

collector = SyncDataCollector(env, stoch_policy, frames_per_batch=50, total_frames=20000, reset_at_each_iter=True)
rb = ReplayBuffer(
            storage=LazyTensorStorage(50),
            sampler=SamplerWithoutReplacement(),
            batch_size=1
        )

utd = 16
pbar = tqdm.tqdm(total=1_000_000)
longest = 0

traj_lens = []
for i, data in enumerate(collector):
    print(f"{i}/{collector.total_frames}")
    if i == 0:
        print(
            "Let us print the first batch of data.\nPay attention to the key names "
            "which will reflect what can be found in this data structure, in particular: "
            "the output of the QValueModule (action_values, action and chosen_action_value),"
            "the 'is_init' key that will tell us if a step is initial or not, and the "
            "recurrent_state keys.\n",
            data,
        )
    pbar.update(data.numel())
    # it is important to pass data that is not flattened
    rb.extend(data.unsqueeze(0).to_tensordict().cpu())
    for _ in range(utd):
        s = rb.sample().to(device)
        loss_vals = loss_fn(s)

        loss_vals["loss"].backward()
        optim.step()
        optim.zero_grad()
    longest = max(longest, data["step_count"].max().item())
    pbar.set_description(
        f"steps: {longest}, loss_val: {loss_vals['loss'].item(): 4.4f}, action_spread: {data['action'].sum(0)}"
    )
    stoch_policy.step(data.numel())
    updater.step()

    if i % 50 == 0:
        with set_exploration_type(ExplorationType.MODE), torch.no_grad():
            rollout = env.rollout(10000, stoch_policy)
            traj_lens.append(rollout.get(("next", "step_count")).max().item())