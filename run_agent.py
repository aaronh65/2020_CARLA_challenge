#####################
# this script is used on my local machine
# you need to run CARLA before running this script

import os, sys, time
import yaml
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='devtest', choices=['devtest','testing','training','debug'])
parser.add_argument('--route', type=int, default=1)
parser.add_argument('--agent', type=str, default='image_agent', choices=['image_agent', 'auto_pilot', 'privileged_agent', 'rl_agent'])
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

def mkdir_if_not_exists(_dir):
    if not os.path.exists(_dir):
        print(f"Creating a directory at {_dir}")
        os.makedirs(_dir)

# make base save path + log dir
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.debug:
    save_path_base = f'leaderboard/results/{args.agent}/debug/{date_str}/{args.split}'
else:
    save_path_base = f'leaderboard/results/{args.agent}/{date_str}/{args.split}'

mkdir_if_not_exists(f'{save_path_base}/logs')


# route path
route_prefix = f'leaderboard/data/routes_{args.split}'
route_name = f'route_{args.route:02d}'
route_path = f'{route_prefix}/{route_name}.xml'

# make image + performance plot dirs
if args.save_images:
    save_images_path = f'{save_path_base}/images/{route_name}'
    for rep_number in range(args.repetitions):
        mkdir_if_not_exists(f'{save_images_path}/repetition_{rep_number:02d}')
save_perf_path = f'{save_path_base}/plots/{route_name}'
mkdir_if_not_exists(save_perf_path)
  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SAVE_PATH_BASE"] = save_path_base
os.environ["SAVE_IMAGES"] = "1" if args.save_images else "0"
os.environ["ROUTE_NAME"] = route_name

# agent-specific configurations
weights_path = 'leaderboard/config'
if args.agent == 'auto_pilot':
    config = 'none' # change to anything except 'none' to save training data
elif args.agent == 'image_agent':
    config = '{weights_path}/image_model.ckpt' # NN weights in leaderboard/configs
elif args.agent == 'privileged_agent':
    config = '{weights_path}/map_model.ckpt' # NN weights in leaderboard/configs
elif args.agent == 'rl_agent':
    config_dict = {'mode': 'train', 'world_port': 2000, 'tm_port': 8000}
    config = f'{save_path_base}/config.yml'
    with open(config, 'w') as f:
        yaml.dump(config_dict, f)
else:
    config = 'None'
 
cmd = f'bash run_agent.sh {args.agent} {route_path} {save_path_base} {config} {args.repetitions}'
print(f'running {cmd}')
os.system(cmd)
