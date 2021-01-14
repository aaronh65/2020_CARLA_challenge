import yaml

from team_code.base_agent import BaseAgent
from team_code.carla_env import CarlaEnv
from team_code.env_utils import *
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.env_checker import check_env
from team_code.sac_models import SAC_LB
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from carla import VehicleControl
import numpy as np
np.set_printoptions(precision=2, suppress=True)

def get_entry_point():
    return 'RLAgent'

class RLAgent(BaseAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        
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
        # get experience tuple for replay buffer
        s_prev = self.env.get_last_state() # get last state before you step
        a_prev = self.env.get_last_action()
        r_prev = self.env.compute_reward(s_prev, a_prev)
        s_new, r, done, info = self.env.step(dummy_action)
        s_new_test = s_new.copy()
        add_vec = np.array([-3,0,0,0,90,0])
        s_new_test += add_vec

        s_new = vector_to_transform(s_new)
        s_new_test = vector_to_transform(s_new_test)

        # compute action for the current state and execute 
        target_transform = convert_transform(s, s_new_test)
        print()
        #print(timestamp)
        #print(s.get_matrix())
        #print(s_new_test.get_matrix())
        #print(target_transform)
        #print()

        control = VehicleControl()
        control.steer = 0
        control.throttle = 0.5
        control.brake = False
        return control

    def run_test_step(self, input_data, timestamp):
        control = VehicleControl()
        return control

