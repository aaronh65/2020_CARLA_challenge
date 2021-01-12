import gym
import numpy as np

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from team_code.env_utils import *
from agents.tools.misc import *

class CarlaEnv(gym.Env):

    def __init__(self):
        super(CarlaEnv, self).__init__()


        ''' Observation space
        One vector of length 6 representing location/rotation of
        target waypoint in hero's frame
        '''
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(6,), dtype=np.float32)

        ''' Action space
        One vector of length 2 representing steering/throttle
        '''
        self.action_space = gym.spaces.Box(low=-1000, high=1000, shape=(2,), dtype=np.float32)

    def step(self, action):
        obs = self.provider.get_transform(self.hero)
        obs = transform_to_vector(obs)

        reward = 1
        done = False
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.provider = CarlaDataProvider
        self.world = self.provider.get_world()
        self.map = self.world.get_map()
        self.hero = self.provider.get_hero_actor()

        self.route = self.provider.get_ego_vehicle_route()
        self.route_waypoints = [self.map.get_waypoint(route_elem[0]) for route_elem in self.route]
        self.route_transforms = [waypoint.transform for waypoint in self.route_waypoints]
        self.route_locations = [transform.location for transform in self.route_transforms]
        self.route_rotations = [transform.rotation for transform in self.route_transforms]

        self.last_state = self.provider.get_transform(self.hero)
        self.last_action = (0,0)

        return transform_to_vector(self.last_state)

        #draw_waypoints(self.world, self.route_waypoints)
    def get_last_state(self):
        return self.last_state

    def render(self):
        pass

    def close(self):
        pass
