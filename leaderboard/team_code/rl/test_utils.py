import carla
import numpy as np
from env_utils import *

def add_location(location, dx=0, dy=0, dz=0):
    return carla.Location(
            location.x + dx,
            location.y + dy,
            location.z + dz)

def add_rotation(rotation, dp=0, dy=0, dr=0):
    return carla.Rotation(
            rotation.pitch + dp,
            rotation.yaw + dy,
            rotation.roll + dr)

def add_transform(transform, dx=0, dy=0, dz=0, dp=0, dyaw=0, dr=0):
    location = add_location(transform.location,dx,dy,dz)
    rotation = add_rotation(transform.rotation,dp,dyaw,dr)
    return carla.Transform(location, rotation)



def main(client, world, cmap):
    spawn_points = cmap.get_spawn_points()
    # a transform will map points from its frame to the global frame

    #for i, sp in enumerate(spawn_points):
    #    world.debug.draw_string(sp.location, str(i), life_time=30.0)


    ref = spawn_points[1]
    ref_location = ref.location
    ref_rotation = ref.rotation

    red = carla.Color(255,0,0)
    green = carla.Color(0,255,0)
    blue = carla.Color(0,0,255)

    forward = add_transform(ref, dx=5)

    right = add_transform(ref, dy=5)

    up = add_transform(ref, dz=5)

    forward_pyaw = add_transform(ref, dx=5, dyaw=45)

    _ = convert_transform(forward_pyaw, ref)


    transforms = []
    #transforms.append(forward)
    #transforms.append(right)
    #transforms.append(up)
    transforms.append(forward_pyaw)
    draw_transforms(world, [ref], color=green)
    draw_transforms(world, transforms, color=red)

    transforms

if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    cmap = world.get_map()
    main(client, world, cmap)
