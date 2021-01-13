import carla
import numpy as np
from env_utils import *


def main(client, world, cmap):
    spawn_points = cmap.get_spawn_points()
    # a transform will map points from its frame to the global frame

    #reference = spawn_points[1]
    #for i, sp in enumerate(spawn_points):
    #    world.debug.draw_string(sp.location, str(i), life_time=30.0)


    ref = spawn_points[5]
    ref_location = ref.location
    ref_rotation = ref.rotation

    forward_tf = carla.Transform(
            ref_location + carla.Location(3,0,0),
            ref_rotation)

    transforms = [ref, forward_tf]
    draw_transforms(world, transforms)

    


if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    cmap = world.get_map()
    main(client, world, cmap)
