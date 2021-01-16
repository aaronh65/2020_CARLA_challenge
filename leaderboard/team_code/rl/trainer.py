import signal
import time
import argparse
import traceback

import numpy as np
np.set_printoptions(precision=3, suppress=True)

from carla import Client
from env import CarlaEnv
from stable_baselines.common.env_checker import check_env
from waypoint_agent import WaypointAgent
from leaderboard.utils.route_indexer import RouteIndexer

def get_route_indexer(args, agent):
    route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
    print(agent)
    for ri in range(len(route_indexer._configs_list)):
        route_indexer.get(ri).agent = agent
    return route_indexer

def get_route_config(route_indexer, idx=None, empty=False):
    num_configs = len(route_indexer._configs_list)
    assert num_configs > 0, 'no configs in route indexer'
    if idx is None:
        print('not idx')
        config = route_indexer.get(np.random.randint(num_configs))
    else:
        assert 0 <= idx < num_configs, 'route config index out of range'
        config = route_indexer.get(idx)
    rconfig = {'config': config, 'extra_args': {'empty': empty}}
    return rconfig


def train(args, env, agent):
    
    route_indexer = get_route_indexer(args, agent) 
    rconfig = get_route_config(route_indexer, idx=0, empty=args.empty)
    print(rconfig['config'].town)
    #rconfig = route_indexer.get(np.random.randint(num_configs))
    state = env.reset(rconfig)
    for step in range(args.total_timesteps):

        # randomly explore for a bit
        if step < args.burn_timesteps:
            # act randomly by sampling from action space
            pass
        else:
            # query policy
            pass

        # step environment with action
        action = np.zeros(2)
        obs, reward, done, info = env.step(action)
        #time.sleep(0.05)

        if done or step % 500 == 499:
        #if done:
            print('resetting')
            env.cleanup()
            rconfig = get_route_config(route_indexer, empty=args.empty)
            state = env.reset(rconfig)

        # store in replay buffer

        # train at this timestep if applicable

        # save model if applicable
        

    #print('done training')

def main(args):
    client = Client('localhost', 2000)
    agent_config = {}
    agent_config['mode'] = 'train'
    agent = WaypointAgent(agent_config)
        
    env_config = {}
    env_config['world_port'] = 2000
    env_config['tm_port'] = 8000
    env_config['tm_seed'] = 0

    try:
        env = CarlaEnv(client, agent, env_config)
        print(env)
        #check_env(env)
        train(args, env, agent)
    except KeyboardInterrupt:
        print('caught Ctrl-C')
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
    finally:
        env.cleanup()
        del env
        #client.reload_world()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--routes', type=str)
    parser.add_argument('--scenarios', type=str)
    parser.add_argument('--repetitions', type=int)
    #parser.add_argument('--total_timesteps', type=int, default=1000000)
    #parser.add_argument('--burn_timesteps' , type=int, default=2500)
    parser.add_argument('--total_timesteps', type=int, default=1000)
    parser.add_argument('--burn_timesteps' , type=int, default=25)
    parser.add_argument('--empty', type=bool, default=True)
    args = parser.parse_args()
    #train(args)
    #generate_dense_waypoints(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
