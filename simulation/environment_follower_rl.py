import time
import random
import numpy as np
import pygame
from simulation.connection import carla
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor
from simulation.settings import *
from carla import Transform, Location, Rotation

import torch

from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from torchrl.envs import EnvBase
from torchrl.envs.transforms import Compose
from torchrl.envs.utils import check_env_specs, set_exploration_mode
from autoencoder.encoder import VariationalEncoder






class CarlaEnvironment(EnvBase):

    def __init__(self, client, world, town, traffic_manager, checkpoint_frequency=100, continuous_action=True, obs_size=101, device='cpu') -> None:
        super(CarlaEnvironment, self).__init__()
        self.to(device)
        self.dtype = torch.float64
        self.client = client
        self.world = world
        self.n_agents = 1
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.vehicle_leader = None
        self.settings = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start=True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town
        self.traffic_manager = traffic_manager

        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        self.action_size = 2
        self.obs_size = obs_size
        # self.conv_encoder = VariationalEncoder(95).to(self.device)
        # self.conv_encoder.load()
        # self.conv_encoder.eval()
        # specs
        self.action_spec = BoundedTensorSpec(minimum=-1, maximum=1, shape=torch.Size([self.n_agents,self.action_size])) # limit the action values

        self.image_spec = UnboundedContinuousTensorSpec(shape=torch.Size([self.n_agents, 160, 80, 3]), dtype=self.dtype) # unlimited observation space
        self.nav_spec = UnboundedContinuousTensorSpec(shape=torch.Size([self.n_agents, 6]), dtype=self.dtype) # Navigation spec
        self.observation_spec = CompositeSpec(image=self.image_spec, navigation=self.nav_spec)
        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size([self.n_agents, 1]), dtype=self.dtype) # unlimited reward space(even though we could limit it to (-inf, 0] in this particular example)

        done_spec = DiscreteTensorSpec(2, shape=torch.Size([self.n_agents, 1]), dtype=torch.bool)
        terminated = DiscreteTensorSpec(2, shape=torch.Size([self.n_agents, 1]), dtype=torch.bool)
        self.done_spec = CompositeSpec(done=done_spec, terminated=terminated)
        # self.create_pedestrians()


    # A reset function for reseting our environment.
    def _reset(self, tensordict, **kwargs):
        out_tensordict = TensorDict({}, batch_size=torch.Size(), device=self.device)
        if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.sensor_list.clear()
            self.actor_list.clear()
        self.remove_sensors()
        time.sleep(0.25)
        actors = self.world.get_actors()
        for a in actors:
            if isinstance(a, carla.Vehicle) or isinstance(a, carla.Sensor):
                print('destroying')
                a.destroy()
        time.sleep(0.25)
        # Blueprint of our main vehicle
        vehicle_bp = self.get_vehicle(CAR_NAME)
        if self.town == "Town07":
            transform = self.map.get_spawn_points()[38]  # Town7  is 38
            self.total_distance = 750
        elif self.town == "Town02":
            transform = self.map.get_spawn_points()[1]  # Town2 is 1
            self.total_distance = 780
        else:
            transform = random.choice(self.map.get_spawn_points())
            self.total_distance = 250

        self.vehicle_leader = self.world.spawn_actor(vehicle_bp, transform)
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.map.get_spawn_points()[39])
        if self.fresh_start:
            self.current_waypoint_index = 0
            # Waypoint nearby angle and distance from it
            self.route_waypoints = list()
            self.waypoint = self.map.get_waypoint(self.vehicle_leader.get_location(), project_to_road=True,
                                                  lane_type=(carla.LaneType.Driving))
            current_waypoint = self.waypoint
            self.route_waypoints.append(current_waypoint)
            for x in range(self.total_distance):
                if self.town == "Town07":
                    if x < 650:
                        next_waypoint = current_waypoint.next(1.0)[0]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[-1]
                elif self.town == "Town02":
                    if x < 650:
                        next_waypoint = current_waypoint.next(1.0)[-1]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[0]
                else:
                    next_waypoint = current_waypoint.next(1.0)[0]
                self.route_waypoints.append(next_waypoint)
                current_waypoint = next_waypoint
            # move vehicle leader to waypoint 3
            self.vehicle_leader.set_transform(self.route_waypoints[self.checkpoint_waypoint_index+15].transform)
            self.vehicle.set_transform(self.route_waypoints[self.checkpoint_waypoint_index].transform)
        else:
            # Teleport vehicle to last checkpoint
            transform_leader = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)+15]
            transform_follower = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
            self.vehicle_leader.set_transform(transform_leader.transform)
            self.vehicle.set_transform(transform_follower.transform)
            self.current_waypoint_index = self.checkpoint_waypoint_index



        # Camera Sensor
        self.camera_obj = CameraSensor(self.vehicle)
        while (len(self.camera_obj.front_camera) == 0):
            time.sleep(0.0001)
        self.image_obs = self.camera_obj.front_camera.pop(-1)
        self.sensor_list.append(self.camera_obj.sensor)

        # Third person view of our vehicle in the Simulated env
        if self.display_on:
            self.env_camera_obj = CameraSensorEnv(self.vehicle)
            self.sensor_list.append(self.env_camera_obj.sensor)

        # Collision sensor
        self.collision_obj = CollisionSensor(self.vehicle)
        self.collision_history = self.collision_obj.collision_data
        self.sensor_list.append(self.collision_obj.sensor)

        self.timesteps = 0
        self.rotation = self.vehicle.get_transform().rotation.yaw
        self.previous_location = self.vehicle.get_location()
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.target_speed = 40  # km/h
        self.max_speed = 50.0
        self.min_speed = 15.0
        self.max_distance_from_center = 3
        self.max_distance_from_leader = 25
        self.throttle = float(0.0)
        self.previous_steer = float(0.0)
        self.velocity = float(0.0)
        self.distance_from_center = float(0.0)
        self.angle = float(0.0)
        self.center_lane_deviation = 0.0
        self.distance_covered = 0.0
        l_r = self.vehicle_leader.get_location()
        f_r = self.vehicle.get_location()
        self.lead_dist_obs = np.sqrt((l_r.x-f_r.x)**2 + (l_r.y-f_r.y)**2)
        self.navigation_obs = np.array(
            [[self.throttle, self.velocity, self.previous_steer, self.distance_from_center, self.angle, self.lead_dist_obs]])



        path = [i.transform.location for i in self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)+15:] if not i.is_junction]
        self.vehicle_leader.set_autopilot(True, self.traffic_manager.get_port())
        self.traffic_manager.set_path(self.vehicle_leader, path)
        self.collision_history.clear()
        image_obs = torch.tensor(self.image_obs, dtype=self.dtype).to(self.device)
        image_obs = image_obs.unsqueeze(0)
        navigation_obs = torch.tensor(self.navigation_obs, device=self.device, dtype=self.dtype)
        # image_obs = image_obs.unsqueeze(0)
        # image_obs = image_obs.permute(0,3,2,1)
        # encode_obs = image_obs.flatten()[:95].unsqueeze(0)
        # encode_obs = self.conv_encoder(image_obs)
        # self.obs = torch.cat(((encode_obs, torch.tensor(self.navigation_obs, device=self.device, dtype=self.dtype))), 1)
        out_tensordict.set("image", image_obs)
        out_tensordict.set("navigation", navigation_obs)
                                    # "reward": torch.tensor(0, device=self.device, dtype=self.dtype),
                                    #           "done": torch.tensor(False,
                                    #                                device=self.device)}, batch_size = torch.Size(), device=self.device)
        self.episode_start_time = time.time()
        return out_tensordict

        # except:
        #     self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
        #     self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        #     self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
        #     self.sensor_list.clear()
        #     self.actor_list.clear()
        #     self.remove_sensors()
        #     if self.display_on:
        #         pygame.quit()


# ----------------------------------------------------------------
# Step method is used for implementing actions taken by our agent|
# ----------------------------------------------------------------

    # A step function is used for taking inputs generated by neural net.
    def _step(self, tensordict):
        # try:
        action = tensordict["action"]
        self.timesteps+=1
        self.fresh_start = False
        # Velocity of the vehicle
        velocity = self.vehicle.get_velocity()
        self.velocity = np.sqrt(velocity.x**2 + velocity.y**2) * 3.6
        l_r = self.vehicle_leader.get_location()
        f_r = self.vehicle.get_location()
        self.lead_dist_obs = np.sqrt((l_r.x-f_r.x)**2 + (l_r.y-f_r.y)**2)
        # print(self.lead_dist)

        # Action fron action space for contolling the vehicle with a discrete action
        action = action.detach().cpu().numpy().flatten()
        steer = float(action[0])
        steer = max(min(steer, 1.0), -1.0)
        throttle = float((action[1] + 1.0)/2)
        throttle = max(min(throttle, 1.0), 0.0)
        self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=self.throttle*0.9 + throttle*0.1))
        self.previous_steer = steer
        self.throttle = throttle

        # Traffic Light state
        if self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                traffic_light.set_state(carla.TrafficLightState.Green)

        self.collision_history = self.collision_obj.collision_data

        # Rotation of the vehicle in correlation to the map/lane
        self.rotation = self.vehicle.get_transform().rotation.yaw

        # Location of the car
        self.location = self.vehicle.get_location()


        #transform = self.vehicle.get_transform()
        # Keep track of closest waypoint on the route
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
            if dot > 0.0:
                waypoint_index += 1
            else:
                break

        self.current_waypoint_index = waypoint_index
        # Calculate deviation from center of the lane
        self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
        self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
        self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
        self.center_lane_deviation += self.distance_from_center

        # Get angle difference between closest waypoint and vehicle forward vector
        fwd = self.vector(self.vehicle.get_velocity())
        wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
        self.angle  = self.angle_diff(fwd, wp_fwd)

         # Update checkpoint for training
        if not self.fresh_start:
            if self.checkpoint_frequency is not None:
                self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency


        # Rewards are given below!
        done = False
        reward = 0

        if len(self.collision_history) != 0:
            done = True
            reward = -10
        elif self.distance_from_center > self.max_distance_from_center:
            done = True
            reward = -10
        elif self.episode_start_time + 10 < time.time() and self.velocity < 1.0:
            reward = -10
            done = True
        elif self.velocity > self.max_speed:
            reward = -10
            done = True
        elif self.lead_dist_obs > self.max_distance_from_leader:
            reward = -10
            done = True

        # Interpolated from 1 when centered to 0 when 3 m from center
        centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
        # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
        angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

        if not done:
            if self.continous_action_space:
                if self.velocity < self.min_speed:
                    reward = (self.velocity / self.min_speed) * centering_factor * angle_factor
                elif self.velocity > self.target_speed:
                    reward = (1.0 - (self.velocity-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor
                else:
                    reward = 1.0 * centering_factor * angle_factor
            else:
                reward = 1.0 * centering_factor * angle_factor

        if self.timesteps >= 7500:
            done = True
        elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
            done = True
            self.fresh_start = True
            if self.checkpoint_frequency is not None:
                if self.checkpoint_frequency < self.total_distance//2:
                    self.checkpoint_frequency += 2
                else:
                    self.checkpoint_frequency = None
                    self.checkpoint_waypoint_index = 0

        while(len(self.camera_obj.front_camera) == 0):
            time.sleep(0.0001)

        self.image_obs = self.camera_obj.front_camera.pop(-1)
        normalized_velocity = self.velocity/self.target_speed
        normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
        normalized_angle = abs(self.angle / np.deg2rad(20))
        self.navigation_obs = np.array([[self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center,
                                         normalized_angle, self.lead_dist_obs]])

        # image_obs = torch.tensor(self.image_obs, dtype=torch.float).to(self.device)
        # image_obs = image_obs.unsqueeze(0)
        # image_obs = image_obs.permute(0,3,2,1)
        # # encode_obs = self.conv_encoder(image_obs)
        # encode_obs = image_obs.flatten()[:95].unsqueeze(0)

        image_obs = torch.tensor(self.image_obs, dtype=self.dtype).to(self.device)
        image_obs = image_obs.unsqueeze(0)
        navigation_obs = torch.tensor(self.navigation_obs, device=self.device, dtype=self.dtype)
        # image_obs = image_obs.unsqueeze(0)
        # image_obs = image_obs.permute(0,3,2,1)
        # encode_obs = image_obs.flatten()[:95].unsqueeze(0)
        # encode_obs = self.conv_encoder(image_obs)
        # self.obs = torch.cat(((encode_obs, torch.tensor(self.navigation_obs, device=self.device, dtype=self.dtype))), 1)


        # self.obs = torch.cat(((encode_obs, torch.tensor(self.navigation_obs, device=self.device, dtype=self.dtype))), 1)
        out_tensordict = TensorDict({"image": image_obs,
                                     "navigation": navigation_obs,
                                     "reward": torch.tensor(np.array([[reward]]), device=self.device, dtype=self.dtype),
                                     "done": torch.tensor(np.array([[done]]), device=self.device)}, batch_size=torch.Size(), device=self.device)
        # Remove everything that has been spawned in the env
        if done:
            self.center_lane_deviation = self.center_lane_deviation / self.timesteps
            self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
            self.vehicle_leader.set_autopilot(False)
            time.sleep(0.5)
            self.vehicle.destroy()
            self.vehicle_leader.destroy()
            self.camera_obj.sensor.destroy()
            self.collision_obj.sensor.destroy()
            self.env_camera_obj.sensor.destroy()
        # print(out_tensordict)
        return out_tensordict

        # except Exception as E:
        #     print(E)
        #     self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
        #     self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        #     self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
        #     self.sensor_list.clear()
        #     self.actor_list.clear()
        #     self.remove_sensors()
        #     if self.display_on:
        #         pygame.quit()



# -------------------------------------------------
# Creating and Spawning Pedestrians in our world |
# -------------------------------------------------

    # Walkers are to be included in the simulation yet!
    def create_pedestrians(self):
        try:

            # Our code for this method has been broken into 3 sections.

            # 1. Getting the available spawn points in  our world.
            # Random Spawn locations for the walker
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)

            # 2. We spawn the walker actor and ai controller
            # Also set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find(
                    'controller.ai.walker')
                # Walkers are made visible in the simulation
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # They're all walking not running on their recommended speed
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)

            # set how many pedestrians can cross the road
            #self.world.set_pedestrians_cross_factor(0.0)
            # 3. Starting the motion of our pedestrians
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start()
            # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])

    def _set_seed(self, seed):
        pass
# ---------------------------------------------------
# Creating and Spawning other vehciles in our world|
# ---------------------------------------------------


    def set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot
            # One simple for loop for creating x number of vehicles and spawing them into the world
            for _ in range(0, NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(
                    bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            print("NPC vehicles have been generated in autopilot mode.")
        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])


# ----------------------------------------------------------------
# Extra very important methods: their names explain their purpose|
# ----------------------------------------------------------------

    # Setter for changing the town on the server.
    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)


    # Getter for fetching the current state of the world that simulator is in.
    def get_world(self) -> object:
        return self.world


    # Getter for fetching blueprint library of the simulator.
    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()


    # Action space of our vehicle. It can make eight unique actions.
    # Continuous actions are broken into discrete here!
    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle


    def distance_to_line(self, A, B, p):
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom


    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])


    def get_discrete_action_space(self):
        action_space = \
            np.array([
            -0.50,
            -0.30,
            -0.10,
            0.0,
            0.10,
            0.30,
            0.50
            ])
        return action_space

    # Main vehicle blueprint method
    # It picks a random color for the vehicle everytime this method is called
    def get_vehicle(self, vehicle_name):
        blueprint = self.world.get_blueprint_library().filter(vehicle_name)[0]

        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint


    # Spawn the vehicle in the environment
    def set_vehicle(self, vehicle_bp, spawn_points):
        # Main vehicle spawned into the env
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)


    # Clean up method
    def remove_sensors(self):
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None

if __name__ == '__main__':
    from encoder_init import EncodeState
    from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
    from tensordict.nn.distributions import NormalParamExtractor
    from tensordict.nn import TensorDictModule
    from torchrl.collectors import SyncDataCollector
    from torchrl.data.replay_buffers import ReplayBuffer
    from torchrl.data.replay_buffers.storages import LazyTensorStorage
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.objectives import ClipPPOLoss, ValueEstimators
    from tqdm import tqdm
    from networks import ppo
    from torchrl.envs import TransformedEnv, ObservationNorm, Compose, DoubleToFloat, StepCounter
    from simulation.encoder_transform import EncodeImage
    from autoencoder.encoder import VariationalEncoder
    from loading import load_yaml
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    actors = world.get_actors()
    for a in actors:
        if isinstance(a, carla.Vehicle) or isinstance(a, carla.Sensor):
            a.destroy()
    time.sleep(0.5)
    traffic_manager = client.get_trafficmanager(8000)
    town = 'Town07'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CarlaEnvironment(client, world, town, traffic_manager=traffic_manager, device=device)
    conv_encoder = VariationalEncoder(95).to(device)

    conv_encoder.load()
    conv_encoder.eval()
    env = TransformedEnv(
        env,
        Compose(
            # normalize observations
            EncodeImage(in_keys=["image", 'navigation'], out_keys=["observation"], encoder=conv_encoder, del_keys=False),
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(in_keys=["observation"], ),
            StepCounter()
        )
    )
    env.transform[1].init_stats(num_iter=20, reduce_dim=0, cat_dim=0)

    # config["loc"] = env.transform[0].loc
    # config["scale"] = env.transform[0].scale
    # config = load_yaml("config.yaml")
    # config["frames_per_batch"] = config["frames_per_batch_init"] // config["frame_skip"]  # how many frames to take from the environment
    # config["total_frames"] = config["total_frames_init"] // config["frame_skip"]  # total number of frames to get from the environment
    # # wandb.login()
    # # wandb.init(project=config["env_name"] + '_' + config["algorithm"])  # log using weights and biases
    #
    # torch.manual_seed(config['random_seed'])
    # np.random.seed(config['random_seed'])
    # random.seed(config['random_seed'])
    # env.set_seed(config['random_seed'])
    #
    #
    # check_env_specs(env)
    #
    # # rollout = env.rollout(7)
    # # print("\nrollout of three steps:\n", rollout)
    # # print("\nShape of the rollout TensorDict:\n", rollout.batch_size)
    # actor_net, value_net, logs = ppo.train(config, env, verbose=True)
    share_parameters_policy = True
    # Sampling
    frames_per_batch = 2000  # Number of team frames collected per training iteration
    n_iters = 10  # Number of sampling and training iterations
    total_frames = frames_per_batch * n_iters
    # Training
    num_epochs = 10  # Number of optimization steps per training iteration
    minibatch_size = 80  # Size of the mini-batches in each optimization step
    lr = 3e-4  # Learning rate
    max_grad_norm = 1.0  # Maximum norm for the gradients

    # PPO
    clip_epsilon = 0.2  # clip value for PPO loss
    gamma = 0.9  # discount factor
    lmbda = 0.9  # lambda for generalised advantage estimation
    entropy_eps = 1e-4  # coefficient of the entropy term in the PPO loss
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["observation"].shape[
                -1
            ],  # n_obs_per_agent
            n_agent_outputs=2 * env.action_spec.shape[-1],  # 2 * n_actions_per_agents
            n_agents=1,
            centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
            share_params=share_parameters_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
        # this will just separate the last dimension into two outputs: a loc and a non-negative scale
    )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("observation")],
        out_keys=[("loc"), ("scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=[("loc"), ("scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.low,
            "max": env.action_spec.space.high,
        },
        return_log_prob=True,
        log_prob_key=("sample_log_prob"),
    )  # we'll need the log-prob for the PPO loss

    share_parameters_critic = True
    mappo = True  # IPPO if False

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["observation"].shape[-1],
        n_agent_outputs=1,  # 1 value per agent
        n_agents=1,
        centralised=mappo,
        share_params=share_parameters_critic,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("observation")],
        out_keys=[("state_value")],
    )

    collector = SyncDataCollector(
        env,
        policy,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            frames_per_batch, device=device
        ),  # We store the frames_per_batch collected at each iteration
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,  # We will sample minibatches of this size
    )

    loss_module = ClipPPOLoss(
        actor=policy,
        critic=critic,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_eps,
        normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
    )
    loss_module.set_keys(  # We have to tell the loss where to find the keys
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("sample_log_prob"),
        value=("state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("done"),
        terminated=("terminated"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
    )  # We build GAE
    GAE = loss_module.value_estimator

    optim = torch.optim.Adam(loss_module.parameters(), lr)

    # pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

    episode_reward_mean_list = []
    for i, tensordict_data in enumerate(collector):
        print(i)
        tensordict_data.set(
            ("next", "done"),
            tensordict_data.get(("next", "done"))
        )
        tensordict_data.set(
            ("next", "terminated"),
            tensordict_data.get(("next", "terminated"))
        )
        # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_params,
                target_params=loss_module.target_critic_params,
            )  # Compute GAE and add it to the data

        data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
        replay_buffer.extend(data_view)

        for _ in range(num_epochs):
            for _ in range(frames_per_batch // minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)

                loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                )
                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_grad_norm
                )  # Optional

                optim.step()

                optim.zero_grad()
        collector.update_policy_weights_()
        # Logging
        # done = tensordict_data.get(("next", "done"))
        # episode_reward_mean = (
        #     tensordict_data.get(("next", "reward"))[done].mean().item()
        # )
        # episode_reward_mean_list.append(episode_reward_mean)
        # pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
        # pbar.update()
    check_env_specs(env)
    n_rollout_steps = 10000
    rollout = env.rollout(n_rollout_steps)
    print("rollout of three steps:", rollout)
    print("Shape of the rollout TensorDict:", rollout.batch_size)

