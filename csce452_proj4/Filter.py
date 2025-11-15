import rclpy
from rclpy.node import Node
from Particle import Particle
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Twist


class ParticleFilter(Node):
    def __init__(self):
        self.particles: list[Particle] = {}
        self.map_pub = self.create_publisher(OccupancyGrid, '/floor', 10)
        #Get map from corresponding file (passed as parameter)
        self.declare_parameter('world_file', '')
        self.est_pose = self.create_publisher(Pose2D, '/estimated_pose', 10)
        #Subscribe to /cmd_vel topic
        self.acrtion_msgs = self.create_subscription(Twist, '/cmd_vel', self.getAction, 10)
        #Sub to /floor_sensor topic
        self.obs_msgs = self.create_subscription(null,'.floor_sensor', self.getObservation, 10)
        pass 

    def pubMap(self):
        value = self.get_parameter('world_file').get_parameter_value().string_value

        #Parse world file using a YAML library
        #Format as occupancy grid and publish
        msg = OccupancyGrid()

        self.map_pub.publish(msg)
        pass
    
    def pubBestPosition(self):
        msg = Pose2D()
        self.est_pose.publish(msg)
        pass

    def getObservation(self, msg):

        #After getting the new observation, reweight each particle
        self.reweight()
        pass 

    def getAction(self, msg):

        #After getting the new action, forward projection each particle
        self.forwardProjection()
        pass

    def forwardProjection(self):
        #If projects is out of bounds of the map, particle is invalid
        # Otherwise, model with gaussian noise??
        pass 

    def reweight(self):

        #After reweighting all particles, resample them
        self.resample()
        pass
    
    def resample(self):
        #Choose particles to keep with probability = weight of particle
        pass
