import os, sys, time
import subprocess
import argparse
import traceback
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='testing', choices=['devtest','testing','training','debug'])
parser.add_argument('--route', type=str, default='00')
parser.add_argument('--agent', type=str, default='image_agent', choices=['image_agent', 'auto_pilot'])
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.agent == 'auto_pilot':
    config = 'none'
elif args.agent == 'image_agent':
    config = 'image_model.ckpt'

def mkdir_if_not_exists(_dir):
    if not os.path.exists(_dir):
        print(f"Creating a directory at {_dir}")
        os.makedirs(_dir)

date_str = datetime.now().strftime("%m%d%Y_%H%M")
if args.debug:
    log_dir = f'leaderboard/results/{args.agent}/debug/{date_str}/{args.split}'
else:
    log_dir = f'leaderboard/results/{args.agent}/{date_str}/{args.split}'
mkdir_if_not_exists(f'{log_dir}/logs')
if args.save_images:
    mkdir_if_not_exists(f'{log_dir}/images')
route = f'leaderboard/data/routes_{args.split}/route_{args.route}.xml'

# directly log from command
cmd = f'CUDA_VISIBLE_DEVICES=0 bash run_agent.sh {args.agent} {route} {log_dir} {config} {int(args.save_images)}'
print(f'running {cmd}')
os.system(cmd)
