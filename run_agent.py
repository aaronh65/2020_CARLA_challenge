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
args = parser.parse_args()
route = f'route_{args.route:02d}'

if args.agent == 'auto_pilot':
    config = 'none'
elif args.agent == 'image_agent':
    config = 'image_model.ckpt'

def mkdir_if_not_exists(_dir):
    if not os.path.exists(_dir):
        print(f"Creating a directory at {_dir}")
        os.makedirs(_dir)

# log dir
date_str = datetime.now().strftime("%Y%m%d_%H%M")
if args.debug:
    log_dir = f'leaderboard/results/{args.agent}/debug/{date_str}/{args.split}'
else:
    log_dir = f'leaderboard/results/{args.agent}/{date_str}/{args.split}'
mkdir_if_not_exists(f'{log_dir}/logs')


save_perf_path = f'{log_dir}/plots/{route}'
mkdir_if_not_exists(save_perf_path)
save_images_path = f'{log_dir}/images/{route}'
for rep_number in range(args.repetitions):
    if args.save_images:
        mkdir_if_not_exists(f'{save_images_path}/repetition_{rep_number:02d}')
    
# route path
route_path = f'leaderboard/data/routes_{args.split}/{route}.xml'

cmd = f'bash run_agent.sh {args.agent} {route_path} {log_dir} {config} {args.repetitions}'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SAVE_IMAGES"] = "1" if args.save_images else "0"
os.environ["SAVE_IMAGES_PATH"] = save_images_path
os.environ["SAVE_PERF_PATH"] = save_perf_path
print(f'running {cmd}')
os.system(cmd)
