import carla
import argparse
import time
import traceback

from waypoint_agent import WaypointAgent
from agents.tools.misc import *
from env import CarlaEnv
from env_utils import *
from stable_baselines.common.env_checker import check_env

from leaderboard.utils.route_indexer import RouteIndexer


def train(args, env, agent):
    
    route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)

    rconfig = route_indexer.get(0)
    rconfig.agent = agent
    # setup stopping parameters and metrics
    env.load_world_and_scenario(rconfig)
    
    # loop until target number of interactions
    print(f'training for {args.total_timesteps} timesteps')
    for step in range(args.total_timesteps):

        # randomly explore for a bit
        if step < args.burn_timesteps:
            # act randomly by sampling from action space
            pass
        else:
            # query policy
            pass

        # step environment with action

        # reset environment if done

        # store in replay buffer

        # train at this timestep if applicable

        # save model if applicable
        
        break

    print('done training')

def main(args):
    client = carla.Client('localhost', 2000)
    env_args = {}
    env_args['world_port'] = 2000
    env_args['tm_port'] = 8000
    env_args['tm_seed'] = 0
    env = CarlaEnv(client, env_args)
    #check_env(env)

    agent_config = {}
    agent_config['mode'] = 'train'
    agent = WaypointAgent(agent_config)

    try:
        train(args, env, agent)
    except Exception as e:
        traceback.print_exc(e)
    env.cleanup()
    client.reload_world()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--routes', type=str)
    parser.add_argument('--scenarios', type=str)
    parser.add_argument('--repetitions', type=int)
    #parser.add_argument('--total_timesteps', type=int, default=1000000)
    #parser.add_argument('--burn_timesteps' , type=int, default=2500)
    parser.add_argument('--total_timesteps', type=int, default=100)
    parser.add_argument('--burn_timesteps' , type=int, default=25)
    args = parser.parse_args()
    #train(args)
    #generate_dense_waypoints(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
