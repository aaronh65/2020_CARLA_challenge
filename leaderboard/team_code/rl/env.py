import signal
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

    def __init__(self, client, agent, env_config):
        super(CarlaEnv, self).__init__()

        # state space is carla.Transform represented as a six vector
        # action space is steering angle/throttle
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # setup client and data provider
        self.client = client
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager(env_config['tm_port'])
        self.traffic_manager.set_random_device_seed(env_config['tm_seed'])
        
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(env_config['tm_port'])

        self.scenario = None
        self.manager = ScenarioManager(60, False)
        self.agent_instance = agent
        self.hero_actor = None
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        if self.manager:
            self.manager.signal_handler(signum, frame)
        raise KeyboardInterrupt

    def reset(self, rconfig=None):
        if not rconfig:
            print('Warning! No configuration given - reloading world')
            self.client.reload_world()
            return np.zeros(6)

        # reset world/scenario, get route and start information
        self._load_world_and_scenario(rconfig)
        self._get_hero_route(draw=True)
        start = transform_to_vector(
                self.map.get_waypoint(self.route[0][0]).transform)

        # prepare manager for run
        self.manager._running = True
        self.manager._watchdog.start()

        return start

    # convert action to vehicle control and tick scenario
    def step(self, action):
        if self.manager._running:
            timestamp = None
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp
            if timestamp:
                obs, reward, done, info = self._tick(timestamp)
            return obs, reward, done, info

        else:
            return np.zeros(6), 1, True, {'running': False}

    def _tick(self, timestamp):
        info = {}

        self.manager._tick_scenario(timestamp)
        hero_transform = CarlaDataProvider.get_transform(self.hero_actor)
        obs = transform_to_vector(hero_transform)

        # find target waypoint
        targets = closest_aligned_transform(
                hero_transform, 
                self.route_transforms, 
                self.forward_vectors)

        if len(targets) == 0:
            reward = 0
            done = True
            return obs, reward, done, info

        waypoints = [self.route_waypoints[i] for i in targets]
        locations = [wp.transform.location for wp in waypoints]
        for arrow_end in locations:
            draw_arrow(self.world, hero_transform.location, arrow_end, color=(0,0,255), z=3, life_time=0.05, size=0.5)

        closest = targets[0]
        reward, done = self._get_reward(hero_transform, self.route_transforms[closest])

        return obs, reward, done, info

    def cleanup(self):

        #if self.manager and self.manager.get_running_status():
        if self.manager:
            self.manager.cleanup()
            if self.manager._watchdog._timer:
                self.manager._watchdog.stop()

            if self.manager.get_running_status():
                if self.manager.scenario:
                    self.manager.scenario.terminate()

                if self.manager._agent:
                    self.manager._agent.cleanup()
                    self.manager._agent = None

        if self.scenario:
            self.scenario.remove_all_actors()
            self.scenario = None

        # Simulation still running and in synchronous mode?
        if self.manager and self.manager.get_running_status() and self.world:
            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        CarlaDataProvider.cleanup()
        self.hero_actor = None

        if self.agent_instance:
            # just clears sensor interface for resetting
            self.agent_instance.destroy() 

    def __del__(self):
        #self.cleanup()
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world
        if hasattr(self, 'scenario') and self.scenario:
            del self.scenario



    def _load_world_and_scenario(self, rconfig):
        config = rconfig['config']
        extra_args = rconfig['extra_args']

        # setup world and retrieve map
        self.world = self.client.load_world(config.town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / 20.0
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.world.reset_all_traffic_lights()
        self.map = self.world.get_map()

        # setup provider and tick to check correctness
        CarlaDataProvider.set_world(self.world)
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()
        
        # setup scenario and scenario manager
        self.scenario = RouteScenario(
                self.world, 
                config, 
                criteria_enable=False, 
                extra_args=extra_args)
        self.manager.load_scenario(
                self.scenario, 
                config.agent,
                config.repetition_index)
        self.hero_actor = CarlaDataProvider.get_hero_actor()
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

     

    def _get_hero_route(self, draw=False):
        # retrieve new hero route
        self.map = self.world.get_map()
        self.route = CarlaDataProvider.get_ego_vehicle_route()
        route_locations = [route_elem[0] for route_elem in self.route]
        self.route_waypoints = [self.map.get_waypoint(loc) 
                for loc in route_locations]
        self.route_transforms = np.array([transform_to_vector(wp.transform) 
                for wp in self.route_waypoints])
        forward_vectors = [wp.transform.get_forward_vector() for wp in self.route_waypoints]
        self.forward_vectors = np.array([[v.x, v.y, v.z] for v in forward_vectors])
        if draw:
            draw_waypoints(self.world, self.route_waypoints, life_time=100)

    def _get_reward(self, hero, closest):

        pass

    def render(self):
        pass

    def close(self):
        pass
