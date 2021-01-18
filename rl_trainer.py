import os, sys, time
import yaml
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='training', choices=['debug', 'devtest', 'testing', 'training'])
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

def mkdir_if_not_exists(_dir):
    if not os.path.exists(_dir):
        print(f"Creating a directory at {_dir}")
        os.makedirs(_dir)

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
prefix = '/home/aaron/workspace/carla/2020_CARLA_challenge'
rpath = 'leaderboard/results/rl/waypoint_agent'
suffix = f'debug/{date_str}' if args.debug else f'{date_str}' 
base_save_path = f'{prefix}/{rpath}/{suffix}'
mkdir_if_not_exists(f'{base_save_path}/weights')
if args.save_images:
    mkdir_if_not_exists(f'{base_save_path}/images')

#os.environ["SAVE_IMAGES"] = args.save_images

env_config = {
        'world_port': 2000,
        'trafficmanager_port': 8000,
        'trafficmanager_seed': 0,
        'route_split': args.split,
        'empty': True,
        'save_images': args.save_images,
        }

sac_config = {
        'mode': 'train',
        'total_timesteps': 500000,
        'burn_timesteps': 2000,
        #'total_timesteps': 1000,
        #'burn_timesteps': 100,
        'train_frequency': 1,
        'gradient_steps': 1,
        'target_update_interval': 1,
        'save_frequency': 1000,
        'log_frequency': 1000,
        'verbose': args.verbose,
        'save_images': args.save_images,
        }

config = {'env_config': env_config, 'sac_config': sac_config}
config_path = f'{base_save_path}/config.yml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

os.environ["BASE_SAVE_PATH"] = base_save_path
os.environ["ROUTE_SPLIT"] = args.split
os.environ["CONFIG_PATH"] = config_path

cmd = f'bash scripts/rl_trainer.sh'
os.system(cmd)
