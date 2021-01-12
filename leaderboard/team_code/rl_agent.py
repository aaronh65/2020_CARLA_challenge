import yaml

from team_code.base_agent import BaseAgent
from team_code.carla_env import CarlaEnv
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.env_checker import check_env
from team_code.sac_models import SAC_LB
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from carla import VehicleControl
import numpy as np

def get_entry_point():
    return 'RLAgent'

class RLAgent(BaseAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        
        # create config
        with open(self.config_path, 'r') as f:
            self.config = yaml.load(f)

        #class Bunch(object):
        #    def __init__(self, adict):
        #        self.__dict__.update(adict)
        #self.config = Bunch(self.config)

        
    def _init(self):
        super()._init()
        self.env = CarlaEnv()
        self.env.reset()
        check_env(self.env)
        self.model = SAC_LB(MlpPolicy, self.env)

    def tick(self, input_data):
        result = super().tick(input_data)
        return result

    def run_step(self, input_data, timestamp):
        
        if self.config['mode'] == 'train':
            control = self.run_train_step(input_data, timestamp)
        elif self.config['mode'] == 'test':
            control = self.run_test_step(input_data, timestamp)
        else:
            control = VehicleControl()
            control.steer = 0
            control.throttle = 1.0
            control.brake = False
        
        return control

    def run_train_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)

        dummy_action = (0,0)
        s = self.env.get_last_state()
        s_new, r, done, info = self.env.step(dummy_action)

        control = VehicleControl()
        control.steer = 0
        control.throttle = 0.5
        control.brake = False
        return control

    def run_test_step(self, input_data, timestamp):
        control = VehicleControl()
        return control

