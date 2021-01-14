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


def train(args, env):
    
    route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
    agent_config = {}
    agent_config['mode'] = 'train'
    agent = WaypointAgent(agent_config)

    rconfig = route_indexer.get(0)
    rconfig.agent = agent
    rconfig.empty = True
    extra_args = {}
    extra_args['empty'] = True

    # setup stopping parameters and metrics
    scenario_args = {'rconfig': rconfig, 'extra_args': extra_args}
    
    state = env.reset(scenario_args)
    print(state)
    
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
        action = np.zeros(2)
        obs, reward, done, info = env.step(action)

        # reset environment if done
        if (step+1) % 50 == 0:
            done = True
            rconfig = route_indexer.get(1)
            agent = WaypointAgent(agent_config)
            rconfig.agent = agent
            scenario_args['rconfig'] = rconfig

        if done:
            state = env.reset(scenario_args)
            print(state)

        # store in replay buffer

        # train at this timestep if applicable

        # save model if applicable
        

    print('done training')

def main(args):
    client = carla.Client('localhost', 2000)
    env_args = {}
    env_args['world_port'] = 2000
    env_args['tm_port'] = 8000
    env_args['tm_seed'] = 0
    env = CarlaEnv(client, env_args)
    #check_env(env)

    
    try:
        train(args, env)
    except Exception as e:
        try: 
            traceback.print_exc(e)
        except Exception as ne:
            print('could not traceback error because')
            print(ne)
            print('original error is')
            print(e)
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
