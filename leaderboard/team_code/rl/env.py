import gym
import numpy as np

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from env_utils import *
from agents.tools.misc import *

class CarlaEnv(gym.Env):

    def __init__(self, client, env_args):
        super(CarlaEnv, self).__init__()

        # state space is carla.Transform represented as a six vector
        # action space is steering angle/throttle
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # setup client and data provider
        self.client = client
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager(env_args['tm_port'])
        self.traffic_manager.set_random_device_seed(env_args['tm_seed'])
        
        self.provider = CarlaDataProvider
        self.provider.set_client(self.client)
        self.provider.set_world(self.world)
        self.provider.set_traffic_manager_port(env_args['tm_port'])

        self.scenario = None
        self.manager = ScenarioManager(60, True)

    # convert action to vehicle control and tick scenario
    def step(self, action):
        obs = np.zeros(6)

        reward = 1
        done = False
        info = {}

        return obs, reward, done, info

    def _load_world_and_scenario(self, config):
        rconfig = config['rconfig']
        extra_args = config['extra_args']

        # setup world and provider
        self.world = self.client.load_world(rconfig.town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / 20.0
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.world.reset_all_traffic_lights()
        self.map = self.world.get_map()

        self.provider.set_world(self.world)
        if self.provider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()
        
        # setup scenario and scenario manager
        self.scenario = RouteScenario(self.world, rconfig, criteria_enable=False, extra_args=extra_args)
        self.manager.load_scenario(self.scenario, rconfig.agent, rconfig.repetition_index)

    def reset(self, config=None):

        if not config:
            print('Warning! No configuration given - reloading world')
            self.client.reload_world()
            return np.zeros(6)

        self._load_world_and_scenario(config)

        self.route = self.provider.get_ego_vehicle_route()
        start_waypoint = self.map.get_waypoint(self.route[0][0])
        state = transform_to_vector(start_waypoint.transform)

        return state


    def get_hero_route(self, draw=False):
        # retrieve new hero route
        self.map = self.world.get_map()
        self.route_waypoints = [self.map.get_waypoint(route_elem[0]) for route_elem in self.route]
        if draw:
            draw_waypoints(self.world, self.route_waypoints)


    def cleanup(self):

        if self.manager.get_running_status():
            if self.manager.scenario is not None:
                self.manager.scenario.terminate()

            if self.manager._agent is not None:
                self.manager._agent.cleanup()
                self.manager._agent = None

        if self.scenario:
            self.scenario.remove_all_actors()
            self.scenario = None

        # Simulation still running and in synchronous mode?
        if self.manager and self.manager.get_running_status() \
                and hasattr(self, 'world') and self.world:

            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        self.provider.cleanup()
        if hasattr(self, 'hero') and self.hero:
            self.hero.destroy()
            self.hero = None

    def render(self):
        pass

    def close(self):
        pass
