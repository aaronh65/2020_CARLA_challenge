#####################
# this script is used on my local machine
# you need to run CARLA before running this script

import os, sys, time
import yaml
import argparse
from datetime import datetime
from leaderboard.team_code.common.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='devtest', choices=['devtest','testing','training','debug'])
parser.add_argument('--route', type=int, default=3)
parser.add_argument('--agent', type=str, default='lbc/image_agent', choices=['lbc/image_agent', 'lbc/auto_pilot', 'lbc/privileged_agent', 'rl/waypoint_agent'])
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

# make base save path + log dir
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
end_str = f'debug/{date_str}/{args.split}' if args.debug else f'{date_str}/{args.split}' 
base_save_path = f'leaderboard/results/{args.agent}/{end_str}'
mkdir_if_not_exists(f'{base_save_path}/logs')

# route path
route_path = f'leaderboard/data/routes_{args.split}'
route_name = f'route_{args.route:02d}'
route_path = f'{route_path}/{route_name}.xml'

# make image + performance plot dirs
if args.save_images:
    save_images_path = f'{base_save_path}/images/{route_name}'
    for rep_number in range(args.repetitions):
        mkdir_if_not_exists(f'{save_images_path}/repetition_{rep_number:02d}')
save_perf_path = f'{base_save_path}/plots/{route_name}'
mkdir_if_not_exists(save_perf_path)

# agent-specific configurations
config = {}
config['save_images'] = args.save_images
conda_env = 'lb'
if args.agent == 'common/straight_agent':
    pass
elif args.agent == 'lbc/auto_pilot':
    config['save_data'] = False
elif args.agent == 'lbc/image_agent':
    conda_env = 'lblbc'
    config['weights_path'] = 'leaderboard/config/image_model.ckpt'
elif args.agent == 'lbc/privileged_agent':
    conda_env = 'lblbc'
    config['weights_path'] = 'leaderboard/config/map_model.ckpt'
elif args.agent == 'rl/waypoint_agent':
    conda_env = 'lbrl'
    config['mode'] = 'train'
    config['world_port'] = 2000
    config['tm_port'] = 8000

config_path = f'{base_save_path}/config.yml'
with open(config_path, 'w') as f:
    yaml.dump(config, f)

# environ variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CONDA_ENV"] = conda_env
os.environ["BASE_SAVE_PATH"] = base_save_path
os.environ["ROUTE_NAME"] = route_name
os.environ["WORLD_PORT"] = "2000"
os.environ["TM_PORT"] = "8000"
 
cmd = f'bash scripts/run_agent.sh {args.agent} {config_path} {route_path} {args.repetitions}'
print(f'running {cmd}')
os.system(cmd)
