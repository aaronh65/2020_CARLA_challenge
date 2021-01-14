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

def load_and_wait_for_world(config, world, ego_vehicles=None):
    settings = world.get_settings()
    settings.fixed_delta_seconds = 1.0 / 20
    settings.synchronous_mode = True

def generate_dense_waypoints(args):
    from leaderboard.utils.route_manipulation import interpolate_trajectory
    root = '/home/aaron/workspace/carla/2020_CARLA_challenge'
    split = args.routes.split('/')[-1]
    split = split.split('_')[1].split('.')[0]

    client = carla.Client('localhost', 2000)
    
    route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
    while route_indexer.peek():
        config = route_indexer.next()
        world = client.load_world(config.town)
        cmap = world.get_map()
        gps_route, route = interpolate_trajectory(world, config.trajectory)
        route_vectors = np.asarray([transform_to_vector(elem[0]) for elem in route])
        #route = convert_transform_to_location(route)
        #waypoints = [cmap.get_waypoint(elem[0]) for elem in route]
        #draw_waypoints(world, waypoints)

        route_id = int(config.name.split('_')[1])
        route_id = f'{route_id:02d}'
        save_path = f'{root}/leaderboard/data/routes_{split}/route_{route_id}_{config.town}_dense.npy'
        with open(save_path, 'wb') as f:
            np.save(f, route_vectors)
        print(save_path)


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
