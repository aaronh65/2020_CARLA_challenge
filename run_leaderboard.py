import os, sys, time
import subprocess
import argparse
import traceback
import psutil
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='devtest', choices=['devtest','testing','training','debug'])
parser.add_argument('--agent', type=str, default='image_agent', choices=['image_agent', 'auto_pilot'])
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--ssd', type=int, default=0, choices=[0,1])
parser.add_argument('--local', action='store_true')
args = parser.parse_args()

if args.agent == 'auto_pilot':
    config = 'none'
elif args.agent == 'image_agent':
    config = 'image_model.ckpt'

if args.local:
    prefix = '/home/aaron/workspace/carla'
else:
    prefix = '/home/aaronhua'

def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

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

try:
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    if args.debug:
        log_dir = f'leaderboard/results/{args.agent}/debug/{date_str}/{args.split}'
    else:
        log_dir = f'leaderboard/results/{args.agent}/{date_str}/{args.split}'
    if not args.local:
        log_dir = f'/ssd{args.ssd}/aaronhua/{log_dir}'
    mkdir_if_not_exists(f'{log_dir}/logs')

    route_prefix = f'leaderboard/data/routes_{args.split}'
    routes = [f'{route_prefix}/{route}' for route in sorted(os.listdir(route_prefix)) if route.endswith('.xml')]

    if args.debug and args.local:
        routes = routes[24:25]
    
    # launch CARLA servers
    gpus=list(range(args.gpus))
    port_map = {gpu: (get_open_port(), get_open_port()) for gpu in gpus}
    for gpu in gpus:

        # get open world port, trafficmanager port
        wp, tp = port_map[gpu]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = f'{gpu}'
        env["DISPLAY"] = ""
        
        # CARLA command
        cmd = f'bash {prefix}/CARLA_0.9.10.1/CarlaUE4.sh --world-port={wp} -opengl &> {log_dir}/logs/CARLA_G{gpu}.log'
        carla_procs.append(subprocess.Popen(cmd, env=env, shell=True))

        print(f'running {cmd}')

    base_timeout = 3
    timeout = max(args.gpus*base_timeout, 10)
    print(f'Opened {len(gpus)} CARLA servers, warming up for {timeout} seconds')
    time.sleep(timeout)
    
    # False if gpu[index] not in use by LBC
    lbc_procs = []
    gpus_free = [True] * len(gpus)
    gpus_procs = [None] * len(gpus)
    gpus_routes = [-1] * len(gpus)
    routes_done = [False] * len(routes)

    # main testing loop
    while False in routes_done or 'running' in routes_done:

        # check for finished Leaderboard runs
        for i, (free, proc, route_idx) in enumerate(zip(gpus_free, gpus_procs, gpus_routes)):
            if proc and proc.poll() is not None: # check if server is done
                gpus_free[i] = True
                gpus_procs[i] = None
                gpus_routes[i] = -1
                routes_done[route_idx] = True

        # wait and continue if we need to
        if True not in gpus_free or False not in routes_done:
            time.sleep(10)
            continue
        
        # else run new process
        route_idx = routes_done.index(False)
        gpu = gpus_free.index(True)
        wp, tp = port_map[gpu]
        route = routes[route_idx].split('/')[-1].split('.')[0]
        
        save_perf_path = f'{log_dir}/plots/{route}'
        mkdir_if_not_exists(save_perf_path)
        save_images_path = f'{log_dir}/images/{route}'
        for rep_number in range(args.repetitions):
            if args.save_images:
                mkdir_if_not_exists(f'{save_images_path}/repetition_{rep_number:02d}')

        # setup env
        env = os.environ.copy()
        env["LOCAL"] = "1" if args.local else "0"
        env["SAVE_IMAGES"] = "1" if args.save_images else "0"
        env["SAVE_IMAGES_PATH"] = save_images_path
        env["SAVE_PERF_PATH"] = save_perf_path
        env["CUDA_VISIBLE_DEVICES"] = f'{gpu}'

        # run command
        cmd = f'bash {prefix}/2020_CARLA_challenge/run_leaderboard.sh {wp} {routes[route_idx]} {log_dir} {tp} {args.agent} {config} {args.repetitions} {prefix} &> {log_dir}/logs/AGENT_{route}.log'
        lbc_procs.append(subprocess.Popen(cmd, env=env, shell=True))
        gpus_free[gpu] = False
        gpus_procs[gpu] = lbc_procs[-1]
        gpus_routes[gpu] = route_idx
        routes_done[route_idx] = 'running'
        print(f'running {cmd}')

except KeyboardInterrupt:
    print('detected keyboard interrupt')

except Exception as e:
    traceback.print_exc()

print('shutting down processes...')
for proc in carla_procs + lbc_procs:
    try:
        kill(proc.pid)
    except OSError as e:
        print(e)
        continue
print('done')
