import carla
import math
import numpy as np

# represent transforms as a vector of length 6
def transform_to_vector(transform):
    loc = transform.location
    rot = transform.rotation
    return np.array([loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw])

def matrix_transform_to_vector(mtransform):
    pass

# transforms go from global -> local
''' Takes a target transform T2, and converts it 
    into reference transform T1's frame via T2_ref = T2^-1
'''
def convert_transform(reference, target):
    pass

def draw_transforms(world, transforms, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    for tf in transforms:
        begin = tf.location + carla.Location(z=z)
        angle = math.radians(tf.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=100.0)


#def closest_waypoint(self, waypoint):
#        dist_vec = waypoint.transform.location - self.provider.get_transform(self.hero).location
#        dist = [dist_vec.x, dist_vec.y, dist_vec.z]
#        dist = np.linalg.norm(dist)
#        return dist
#
#def aligned_waypoint(self, waypoint):
#    waypoint_rot = waypoint.transform.rotation
#    hero_rot = self.provider.get_transform(self.hero).
