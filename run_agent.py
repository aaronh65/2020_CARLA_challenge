#####################
# this script is run on my local machine
# you need to run CARLA before running this script

import os, sys, time
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='devtest', choices=['devtest','testing','training','debug'])
parser.add_argument('--route', type=int, default=3)
parser.add_argument('--agent', type=str, default='image_agent', choices=['image_agent', 'auto_pilot', 'privileged_agent'])
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.agent == 'auto_pilot':
    config = 'none'
elif args.agent == 'image_agent':
    config = 'image_model.ckpt'
elif args.agent == 'privileged_agent':
    config = 'map_model.ckpt'

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

# image and performance plot dirs
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

cmd = f'bash run_agent.sh {args.agent} {route_path} {save_path_base} {config} {args.repetitions}'
print(f'running {cmd}')
os.system(cmd)
