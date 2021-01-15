import time
import gym
import numpy as np

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from env_utils import *
from reward_utils import closest_aligned_transform
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
        self.agent_object = None
        self.hero_actor = None

    def set_agent(self, agent_object):
        self.agent_object = agent_object

    def tick(self, obs):

        # find target waypoint
        #target_idx = closest_aligned_transform(obs, self.route_transforms, self.forward_vectors)
        candidates = closest_aligned_transform(
                self.hero_transform, 
                self.route_transforms, 
                self.forward_vectors)
        waypoints = [self.route_waypoints[i] for i in candidates]
        locations = [wp.transform.location for wp in waypoints]
        hero_location = self.hero_transform.location
        for arrow_end in locations:
            draw_arrow(self.world, hero_location, arrow_end, color=(0,0,255), z=3, life_time=0.05)
        #waypoints = [self.route_waypoints[i] for i in candidates]
        #draw_waypoints(self.world, waypoints, color=(0,0,255), z=3, life_time=0.05)
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

        self.hero_transform = self.provider.get_transform(self.hero_actor)
        draw_transforms(self.world, [self.hero_transform], color=(0,255,0), z=3, life_time=0.05)
        obs = np.array(transform_to_vector(self.hero_transform))

        reward, done, info = self.tick(obs)
        return obs, reward, done, info

    def _load_world_and_scenario(self, config):
        rconfig = config['rconfig']
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
                self.agent_object, 
                rconfig.repetition_index)
        self.hero_actor = self.provider.get_hero_actor()
        
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
        forward_vectors = [wp.transform.get_forward_vector() for wp in self.route_waypoints]
        self.forward_vectors = np.array([[v.x, v.y, v.z] for v in forward_vectors])
        if draw:
            draw_waypoints(self.world, self.route_waypoints, life_time=100)

    def cleanup(self):

        if self.manager:
            self.manager.cleanup()
            if self.manager._watchdog._timer:
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

        if self.agent_object:
            # just clears sensor interface for resetting
            self.agent_object.destroy() 

    def render(self):
        pass

    def close(self):
        pass
