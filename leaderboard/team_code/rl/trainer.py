import carla
import argparse
import time

from agents.tools.misc import *
from env_utils import *

def train(args):
    client = carla.Client('localhost', 2000)
    world = client.get_world()


    # setup stopping parameters and metrics
    
    # loop until target number of interactions
    print(args.total_timesteps)
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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--routes', type=str)
    parser.add_argument('--scenarios', type=str)
    parser.add_argument('--repetitions', type=int)
    #parser.add_argument('--total_timesteps', type=int, default=1000000)
    #parser.add_argument('--burn_timesteps' , type=int, default=2500)
    parser.add_argument('--total_timesteps', type=int, default=1000)
    parser.add_argument('--burn_timesteps' , type=int, default=250)
    args = parser.parse_args()
    #train(args)
    #generate_dense_waypoints(args)

if __name__ == '__main__':
    main()
