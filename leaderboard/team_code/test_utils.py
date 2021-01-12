import carla
import numpy as np
from env_utils import *

def main(client, world, cmap):
    spawn_points = cmap.get_spawn_points()
    # a transform will map points from its frame to the global frame

    reference = spawn_points[0]
    


if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    cmap = world.get_map()
    main(client, world, cmap)
