import time
import gym
import numpy as np

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from env_utils import *
from test_utils import *
from reward_utils import closest_transform
#from agents.tools.misc import *

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
        self.manager = ScenarioManager(60, False)
        self.agent_instance = None
        self.actor_instance = None

    def tick(self, obs):

        # find target waypoint
        target_idx = closest_transform(obs, self.route_transforms)
        reward = 1
        done = False
        info = {}

        return reward, done, info

    # convert action to vehicle control and tick scenario
    def step(self, action):
        timestamp = None
        snapshot = self.world.get_snapshot()
        if snapshot:
            timestamp = snapshot.timestamp
        if timestamp:
            self.manager._tick_scenario(timestamp)

        state = self.provider.get_transform(self.actor_instance)
        obs = np.array(transform_to_vector(state))

        reward, done, info = self.tick(obs)
        return obs, reward, done, info

    def _load_world_and_scenario(self, config):
        rconfig = config['rconfig']
        if not self.agent_instance:
            self.agent_instance = rconfig.agent
        extra_args = config['extra_args']

        # setup world and retrieve map
        self.world = self.client.load_world(rconfig.town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / 20.0
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.world.reset_all_traffic_lights()
        self.map = self.world.get_map()

        # setup provider and tick to check correctness
        self.provider.set_world(self.world)
        if self.provider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()
        
        # setup scenario and scenario manager
        self.scenario = RouteScenario(
                self.world, 
                rconfig, 
                criteria_enable=False, 
                extra_args=extra_args)
        self.manager.load_scenario(
                self.scenario, 
                self.agent_instance, 
                rconfig.repetition_index)
        self.actor_instance = self.provider.get_hero_actor()
        
    def reset(self, config=None):
        if not config:
            print('Warning! No configuration given - reloading world')
            self.client.reload_world()
            return np.zeros(6)

        self._load_world_and_scenario(config)
        self._get_hero_route(draw=True)
        state = transform_to_vector(
                self.map.get_waypoint(self.route[0][0]).transform)

        #offset_transform = add_transform(start_waypoint.transform, dx=5)
        #draw_transforms(self.world, [offset_transform])
        self.manager._watchdog.start()
        self.manager.start_system_time = time.time()
        self.manager._running = True
        return state


    def _get_hero_route(self, draw=False):
        # retrieve new hero route
        self.map = self.world.get_map()
        self.route = self.provider.get_ego_vehicle_route()
        route_locations = [route_elem[0] for route_elem in self.route]
        self.route_waypoints = [self.map.get_waypoint(loc) 
                for loc in route_locations]
        self.route_transforms = np.array([transform_to_vector(wp.transform) 
                for wp in self.route_waypoints])
        if draw:
            draw_waypoints(self.world, self.route_waypoints)

    def cleanup(self):

        if self.manager:
            self.manager.cleanup()
            self.manager._watchdog.stop()
            if self.manager.scenario is not None:
                self.manager.scenario.terminate()

            if self.manager._agent is not None:
                self.manager._agent.cleanup()
                self.manager._agent = None

            if self.scenario:
                self.scenario.remove_all_actors()
                self.scenario = None

        # Simulation still running and in synchronous mode?
        if self.world:

            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        self.provider.cleanup()

        if self.agent_instance:
            # just clears sensor interface for resetting
            # instance still exists
            self.agent_instance.destroy() 

    def render(self):
        pass

    def close(self):
        pass
