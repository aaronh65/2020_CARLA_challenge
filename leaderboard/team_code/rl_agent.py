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

        self.global_planner_dao = GlobalRoutePlannerDAO(self.map, 2.0)
        self.global_planner = GlobalRoutePlanner(self.global_planner_dao)
        self.global_planner.setup()

        origin_transform, origin_command = self._global_plan_world_coord[0]
        dest_transform, dest_command = self._global_plan_world_coord[1]

        route = self.global_planner.trace_route(origin_transform.location, dest_transform.location)
        print(route)

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
