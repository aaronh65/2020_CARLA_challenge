import os, sys, time
import subprocess
import argparse
import traceback
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='testing', choices=['devtest','testing','training','debug'])
parser.add_argument('--agent', type=str, default='image_agent', choices=['image_agent', 'auto_pilot'])
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.agent == 'auto_pilot':
    config = 'none'
elif args.agent == 'image_agent':
    config = 'image_model.ckpt'


def get_open_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("",0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port

def mkdir_if_not_exists(_dir):
    if not os.path.exists(_dir):
        print(f"Creating a directory at {_dir}")
        os.makedirs(_dir)

carla_procs = list()
lbc_procs = list()

try:
    date_str = datetime.now().strftime("%m%d%Y_%H%M")
    if args.debug:
        log_dir = f'leaderboard/results/{args.agent}/debug/{date_str}/{args.split}'
    else:
        log_dir = f'leaderboard/results/{args.agent}/{date_str}/{args.split}'
    print(log_dir)
    #mkdir_if_not_exists(f'{log_dir}/logs')
    if args.save_images:
        mkdir_if_not_exists(f'{log_dir}/images')
    
    # launch CARLA servers
    gpus=list(range(args.gpus))
    port_map = {gpu: (get_open_port(), get_open_port()) for gpu in gpus}
    for gpu in gpus:

        # get open world port, trafficmanager port
        wp, tp = port_map[gpu]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = f'{gpu}'
        env["DISPLAY"] = ""
        
        # directly log from command
        cmd = f'bash /home/aaronhua/CARLA_0.9.10.1/CarlaUE4.sh --world-port={wp} &> {log_dir}/logs/CARLA_G{gpu}.log'
        #carla_procs.append(subprocess.Popen(cmd, env=env, shell=True))

        print(f'running {cmd}')

    base_timeout = 3
    timeout = min(args.gpus*base_timeout, 10)
    print(f'Opened {len(gpus)} CARLA servers, warming up for {timeout} seconds')
    time.sleep(timeout)
    
    # False if gpu[index] not in use by LBC
    open_gpus = [True] * len(gpus)
    route_prefix = f'leaderboard/data/routes_{args.split}'
    routes = [f'{route_prefix}/{route}' for route in os.listdir(route_prefix)]
    routes_done = [False] * len(routes)

    # main testing loop
    while False in routes_done or 'running' in routes_done:

        # if all servers busy, wait for a bit
        if True not in open_gpus: 
            time.sleep(5)
            # check each server index, (process, routes index)
            for si, (proc, ri)in enumerate(open_gpus):
                if proc.poll() is not None: # check if server is done
                    open_gpus[si] = True
                    routes_done[ri] = True
            continue

        if False not in routes_done:
            time.sleep(10)
            continue

        # else run new process
        ri = routes_done.index(False)
        gpu = open_gpus.index(True)
        wp, tp = port_map[gpu]
        route = routes[ri].split('/')[-1].split('.')[0]
        
        # directly log from command
        cmd = f'bash /home/aaronhua/2020_CARLA_challenge/run_agent_cluster.sh {gpu} {wp} {routes[ri]} {log_dir} {tp} {args.agent} {config} {args.repetitions} {int(args.save_images)} &> {log_dir}/logs/AGENT_{route}.log'
        #cmd = f'bash /home/aaronhua/2020_CARLA_challenge/run_agent_cluster.sh {gpu} {wp} {routes[ri]} {log_dir} {tp} {args.agent} {config}'
        #lbc_procs.append(subprocess.Popen(cmd, shell=True))

        print(f'running {cmd}')
        open_gpus[gpu] = (lbc_procs[-1], ri)
        routes_done[ri] = 'running'

except KeyboardInterrupt:
    pass

except Exception as e:
    traceback.print_exc()
    pass

for i in range(len(carla_procs)):
    carla_procs[i].kill()
for i in range(len(lbc_procs)):
    lbc_procs[i].kill()
