import argparse, sys
from leaderboard.utils.route_manipulation import interpolate_trajectory
from leaderboard.scenarios.route_scenario import RouteScenario, convert_transform_to_location
from leaderboard.utils.route_indexer import RouteIndexer

from env_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='devtest', choices=['devtest', 'testing', 'training'])
args = parser.parse_args()

root = f'/home/aaron/workspace/carla/2020_CARLA_challenge'
routes = f'{root}/leaderboard/data/routes_{args.split}.xml'
scenarios = f'{root}/leaderboard/data/no_traffic_scenarios.json'
repetitions = 1

client = carla.Client('localhost', 2000)

route_indexer = RouteIndexer(routes, scenarios, repetitions)
while route_indexer.peek():
    config = route_indexer.next()
    world = client.load_world(config.town)
    gps_route, route = interpolate_trajectory(world, config.trajectory)
    route_vectors = np.asarray([transform_to_vector(elem[0]) for elem in route])

    
    route_id = int(config.name.split('_')[1])
    route_name = f'route_{route_id:02d}'
    save_path = f'{root}/leaderboard/data/routes_{args.split}/{route_name}_{config.town}_dense.npy'
    with open(save_path, 'wb') as f:
        np.save(f, route_vectors)
    print(save_path)

    # visualize
    #cmap = world.get_map()
    #route = convert_transform_to_location(route)
    #waypoints = [cmap.get_waypoint(elem[0]) for elem in route]
    #draw_waypoints(world, waypoints)



