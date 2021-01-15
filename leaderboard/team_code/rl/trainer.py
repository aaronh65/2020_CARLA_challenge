import carla
import argparse
import time
import traceback
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from env import CarlaEnv
from stable_baselines.common.env_checker import check_env
from waypoint_agent import WaypointAgent
from leaderboard.utils.route_indexer import RouteIndexer


def train(args, env, agent):

    route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
    for ri in range(len(route_indexer._configs_list)):
        route_indexer.get(ri).agent = agent
    rconfig = route_indexer.get(0)
    scenario_args = {'rconfig': rconfig, 'extra_args': {'empty':True}}
    state = env.reset(scenario_args)
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

        if done:
            env.cleanup()
            state = env.reset(scenario_args)
            time.sleep(5)

        # store in replay buffer

        # train at this timestep if applicable

        # save model if applicable
        

    #print('done training')

def main(args):
    client = carla.Client('localhost', 2000)

    agent_config = {}
    agent_config['mode'] = 'train'
    agent = WaypointAgent(agent_config)

    env_args = {}
    env_args['world_port'] = 2000
    env_args['tm_port'] = 8000
    env_args['tm_seed'] = 0
    env = CarlaEnv(client, env_args)
    env.set_agent(agent)
    #check_env(env)

    
    try:
        train(args, env, agent)
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        #try: 
        #    if type(e) == int:
        #        print(e)
        #    else:
        #        traceback.print_exc(e)
        #except Exception as ne:
        #    print('could not traceback error because')
        #    print(ne)
        #    print('original error is')
        #    print(e)
    env.cleanup()
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
    args = parser.parse_args()
    #train(args)
    #generate_dense_waypoints(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
