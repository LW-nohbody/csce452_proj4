import rclpy
from rclpy.node import Node
from Particle import Particle
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Twist
from example_interfaces.msg import UInt8
from geometry_msgs.msg import Pose
import yaml
from builtin_interfaces.msg import Time


class ParticleFilter(Node):
    def __init__(self):
        self.map: OccupancyGrid
        self.particles: list[Particle] = []
        self.map_pub = self.create_publisher(OccupancyGrid, '/floor', 10)
        #Get map from corresponding file (passed as parameter)
        self.declare_parameter('world_file', '')
        self.est_pose = self.create_publisher(Pose2D, '/estimated_pose', 10)
        #Subscribe to /cmd_vel topic
        self.acrtion_msgs = self.create_subscription(Twist, '/cmd_vel', self.getAction, 10)
        #Sub to /floor_sensor topic
        self.obs_msgs = self.create_subscription(UInt8,'.floor_sensor', self.getObservation, 10)
        
        #populate particles evenly over map
        pass 

    def pubMap(self):
        value = self.get_parameter('world_file').get_parameter_value().string_value

        with open(value, 'r') as f:
            map_yaml = yaml.safe_load(f)
        
        reso = map_yaml["resolution"]
        lines = map_yaml["map"].splitlines()
        print(map_yaml["map"]) # DEBUG:REMOVE

        width = len(lines[0])
        height = len(lines)

        #Format as occupancy grid and publish
        msg = OccupancyGrid()
        msg.header.frame_id = 'world'
        msg.header.stamp = Time(sec=0, nanosec=0)

        msg.info.resolution = reso
        msg.info.width = width
        msg.info.height = height
        msg.info.origin = Pose() #Defaults to origin

        map_data = []
        for row in lines:
            for char in row:
                if char == '.':
                    map_data.append(0) #light
                elif char == '#':
                    map_data.append(1) # dark
                else:
                    map_data.append(-1) # unknown

        msg.data = map_data
        self.map = msg

        self.map_pub.publish(msg)
    
    def pubBestPosition(self):
        msg = Pose2D()
        self.est_pose.publish(msg)
        pass

    def getObservation(self, msg:UInt8):
        newObs: int = msg.data
        #After getting the new observation, reweight each particle
        self.reweight(newObs)

        #After reweighting all particles, resample them
        self.resample()
        pass 

    def getAction(self, msg):

        #After getting the new action, forward projection each particle
        self.forwardProjection()
        pass

    def forwardProjection(self):
        #If projects is out of bounds of the map, particle is invalid
        # Otherwise, model with gaussian noise??
        pass 

    def reweight(self, obs: int):
        for i in self.particles:
            i.setWeight(obs)
    
    def resample(self):
        #Choose particles to keep with probability = weight of particle
        pass
