from team_code.base_agent import BaseAgent
from srunner.scenariomanager.carla_data_provider import *
import carla
import yaml

def get_entry_point():
    return 'RLAgent'

class RLAgent(BaseAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        #class Bunch(object):
        #    def __init__(self, adict):
        #        self.__dict__.update(adict)

        # create config
        with open(path_to_conf_file, 'r') as f:
            self.config = yaml.load(f)
        self.provider = CarlaDataProvider

    def run_step(self, input_data, timestamp):

        print(self.provider.get_hero_actor().id)

        control = carla.VehicleControl()
        control.steer = 0
        control.throttle = 0.5
        control.brake = False

        return control
