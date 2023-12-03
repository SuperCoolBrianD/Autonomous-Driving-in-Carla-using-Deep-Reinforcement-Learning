from simulation.environment_follower_rl import CarlaEnvironment
import carla
import time
import numpy as np
import random
# torch imports
import torch
from torchrl.envs import TransformedEnv, ObservationNorm, Compose, DoubleToFloat, StepCounter
from simulation.encoder_transform import EncodeImage
from autoencoder.encoder import VariationalEncoder

from torchrl.envs.utils import check_env_specs
from agent.ppo_agent import PPOAgent
from torchrl.envs.transforms import CatFrames, InitTracker


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
env = CarlaEnvironment(client, world, town, traffic_manager=traffic_manager, device=device)
conv_encoder = VariationalEncoder(95).to(device)

conv_encoder.load()
conv_encoder.eval()

env = TransformedEnv(
    env,
    Compose(
        # normalize observations
        EncodeImage(in_keys=["pixel", 'navigation'], out_keys=["observation"], encoder=conv_encoder, del_keys=False),
        # CatFrames(5, in_keys=["observation"], out_keys=["observation"], dim=-2),
        DoubleToFloat(in_keys=["observation"], ),
        StepCounter(),
        ObservationNorm(in_keys=["observation"]),

    )
)
print(env.specs)
env.transform[-1].init_stats(num_iter=100, reduce_dim=0, cat_dim=0)
check_env_specs(env)
agent = PPOAgent(env, device)

agent.train()