import carla
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

# represent transforms as a vector of length 6
def transform_to_vector(transform):
    loc = transform.location
    rot = transform.rotation
    return [loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw]

def vector_to_transform(vector):
    x,y,z,roll,pitch,yaw = vector
    loc = carla.Location(x,y,z)
    rot = carla.Rotation(roll,pitch,yaw)
    return carla.Transform(loc, rot)

def matrix_transform_to_vector(mtransform):
    pass

def yaw_difference(T1, T2):

    pass

# transforms go from local -> world
''' Takes a target transform T2, and converts it 
    into reference transform T1's frame via T2_ref = T2^-1
'''
def convert_transform(reference, target):
    target_to_world = np.array(target.get_matrix())
    world_to_reference = np.array(reference.get_inverse_matrix())
    target_to_reference = np.matmul(world_to_reference, target_to_world)
    location = target_to_reference[:3, 3]
    print(location)


    #Rmat = target_to_reference[:3, :3]
    #yaw = np.atan2(A[2,1], A[2,0])
    #pitch = np.acos(A[2,2])
    #roll = np.

    #Rmat = target_to_reference[:3,:3]
    #vec = target.get_forward_vector()
    #rotation_vec = np.array([[vec.x, vec.y, vec.z]]).T
    #rotation_vec = np.matmul(Rmat, rotation_vec)
    print(rotation_vec.flatten())

    return target_to_reference

def draw_transforms(world, transforms, color=(255,0,0), z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    r,g,b = color
    ccolor = carla.Color(r,g,b)
    for tf in transforms:
        begin = tf.location + carla.Location(z=z)
        angle = math.radians(tf.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=15.0, color=ccolor)


#def closest_waypoint(self, waypoint):
#        dist_vec = waypoint.transform.location - self.provider.get_transform(self.hero).location
#        dist = [dist_vec.x, dist_vec.y, dist_vec.z]
#        dist = np.linalg.norm(dist)
#        return dist
#
#def aligned_waypoint(self, waypoint):
#    waypoint_rot = waypoint.transform.rotation
#    hero_rot = self.provider.get_transform(self.hero).
