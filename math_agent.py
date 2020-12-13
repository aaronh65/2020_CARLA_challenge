import os, sys, time
import subprocess
import argparse
import traceback
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='devtest', choices=['devtest','testing','training','debug'])
parser.add_argument('--route', type=int, default=3)
parser.add_argument('--agent', type=str, default='image_agent', choices=['image_agent', 'auto_pilot'])
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--save_math', action='store_true')
parser.add_argument('--run_math', action='store_true')
args = parser.parse_args()

if args.agent == 'auto_pilot':
    config = 'none'
elif args.agent == 'image_agent':
    config = 'image_model.ckpt'

def mkdir_if_not_exists(_dir):
    if not os.path.exists(_dir):
        print(f"Creating a directory at {_dir}")
        os.makedirs(_dir)

# base save path
date_str = datetime.now().strftime("%Y%m%d_%H%M")
if args.debug:
    save_path_base = f'leaderboard/results/{args.agent}/debug/{date_str}/{args.split}'
else:
    save_path_base = f'leaderboard/results/{args.agent}/{date_str}/{args.split}'

# log dir
mkdir_if_not_exists(f'{save_path_base}/logs')

# route path
route_prefix = f'leaderboard/data/routes_{args.split}'
route_name = f'route_{args.route:02d}'
route_path = f'{route_prefix}/{route_name}.xml'

# per-route performance plot dirs
save_perf_path = f'{save_path_base}/plots/{route_name}'
mkdir_if_not_exists(save_perf_path)

# image dirs
if args.save_images:
    save_images_path = f'{save_path_base}/images/{route_name}'
    for rep_number in range(args.repetitions):
        mkdir_if_not_exists(f'{save_images_path}/repetition_{rep_number:02d}')
    
# math project
if args.save_math:
    for rep_number in range(args.repetitions):
        mkdir_if_not_exists(f'{save_path_base}/math/{route_name}/repetition_{rep_number:02d}')

os.environ["RUN_MATH"] = "1" if args.run_math else "0"
os.environ["SAVE_MATH"] = "1" if args.save_math else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SAVE_PATH_BASE"] = save_path_base
os.environ["ROUTE_NAME"] = route_name
os.environ["SAVE_IMAGES"] = "1" if args.save_images else "0"
#os.environ["SAVE_IMAGES_PATH"] = save_images_path
#os.environ["SAVE_PERF_PATH"] = save_perf_path

cmd = f'bash run_agent.sh {args.agent} {route_path} {save_path_base} {config} {args.repetitions}'
print(f'running {cmd}')
os.system(cmd)
