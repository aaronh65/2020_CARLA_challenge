import carla
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def to_array(carla_vector):
    v = carla_vector
    return np.array([v.x, v.y, v.z])
# represent transforms as a vector of length 6
def transform_to_vector(transform):
    loc = transform.location
    rot = transform.rotation
    return np.array([loc.x, loc.y, loc.z, rot.pitch, rot.yaw, rot.roll])

def vector_to_transform(vector):
    x,y,z,pitch,yaw,roll = vector
    loc = carla.Location(x,y,z)
    rot = carla.Rotation(pitch,yaw,roll)
    return carla.Transform(loc, rot)

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

def draw_transforms(world, transforms, color=(255,0,0), z=0.5, life_time=0.05):
    r,g,b = color
    ccolor = carla.Color(r,g,b)
    for tf in transforms:
        begin = tf.location + carla.Location(z=z)
        angle = math.radians(tf.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=life_time, color=ccolor)

def draw_waypoints(world, waypoints, color=(255,0,0), z=0.5, life_time=0.05):
    transforms = [wp.transform for wp in waypoints]
    draw_transforms(world, transforms, color, z, life_time=life_time)

def draw_arrow(world, loc1, loc2, color=(255,0,0), z=0.5, life_time=0.05):
    r,g,b = color
    ccolor = carla.Color(r,g,b)
    world.debug.draw_arrow(loc1, loc2, arrow_size=0.3, life_time=life_time, color=ccolor)

