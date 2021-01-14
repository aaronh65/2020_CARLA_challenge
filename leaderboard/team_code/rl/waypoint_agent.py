import yaml

#from team_code.base_agent import BaseAgent
from leaderboard.autoagents import autonomous_agent
from leaderboard.envs.sensor_interface import SensorInterface
from team_code.rl.sac_models import SAC_LB
from stable_baselines.sac.policies import MlpPolicy

from carla import VehicleControl
import numpy as np
np.set_printoptions(precision=2, suppress=True)

def get_entry_point():
    return 'WaypointAgent'

class WaypointAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file=None):
        if path_to_conf_file:
            if type(path_to_conf_file) == str:
                with open(self.config_path, 'r') as f:
                    self.config = yaml.load(f, Loader=yaml.Loader)
            elif type(path_to_conf_file) == dict:
                self.config = path_to_conf_file
        self.track = autonomous_agent.Track.SENSORS

    def sensors(self):
        return [
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                ]
    def destroy(self):
        print('destroying hero')
        if 'mode' in self.config.keys() and self.config['mode'] == 'train':
            self.sensor_interface = SensorInterface()

    def run_step(self, input_data, timestamp):
        control = VehicleControl()
        control.steer = 0
        control.throttle = 0.5
        control.brake = False
        return control

