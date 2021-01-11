import yaml

from team_code.base_agent import BaseAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import *

from carla import VehicleControl

def get_entry_point():
    return 'RLAgent'

class RLAgent(BaseAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        #class Bunch(object):
        #    def __init__(self, adict):
        #        self.__dict__.update(adict)

        # create config
        with open(self.config_path, 'r') as f:
            self.config = yaml.load(f)
        #self.config = Bunch(self.config)
        
    def _init(self):
        super()._init()
        self.provider = CarlaDataProvider
        self.world = self.provider.get_world()
        self.map = self.world.get_map()
        self.actor = self.provider.get_hero_actor()

        self.dao = GlobalRoutePlannerDAO(self.map, 1.0)
        self.grp = GlobalRoutePlanner(self.dao)
        self.grp.setup()

        self.route = self.provider.get_ego_vehicle_route()
        self.route_waypoints = [self.map.get_waypoint(route_elem[0]) for route_elem in self.route]
        draw_waypoints(self.world, self.route_waypoints)

    def tick(self, input_data):
        result = super().tick(input_data)
        return result

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)

        control = VehicleControl()
        control.steer = 0
        control.throttle = 1.0
        control.brake = False

        return control
