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
import random


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
        # sub to compass
        
        # populate particles evenly over map, with the same weight
        # find particles in each column - divide number of particles by width of map (particle in middle of each column)
        # Find column spacing - divide height of map * resolution by number of particles in each column
        # Starting in bottom left, offset first particle alogn the x axis by half a resolution (sets to middle of column)
        # Then add a resolution to the last particle added to row to get the x coord of next particle in row
        # Then from the first particle in row, add the column spacing to move up to next row and repeat
        # Repeat until map filled
        pass 

    def pubMap(self):
        value = self.get_parameter('world_file').get_parameter_value().string_value

        with open(value, 'r') as f:
            map_yaml = yaml.safe_load(f)
        
        reso = map_yaml["resolution"]
        lines = map_yaml["map"].splitlines()

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
        best_weight: float = self.particles[0].weight
        best_particle: Particle = self.particles[0]

        # TODO: Find better method to get best guess position, right now just basing it on highest weight
        for p in self.particles:
            if(p.weight > best_weight):
                best_weight = p.weight
                best_particle = p

        msg:Pose2D = best_particle.state

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
        # forward project movement of particle based on action
        # Then add gaussain noise to final position
        pass 

    def reweight(self, obs: int):
        for p in self.particles:
            # if particle is outside map, force weight to be 0 (particle will never exit map) -> must be removed in resample
            if((p.state.x > (self.map.info.width * self.map.info.resolution))
                or (p.state.x < 0)
                or (p.state.y > (self.map.info.height * self.map.info.resolution)) 
                or (p.state.y < 0)
            ):
                p.weight = 0
            else:
                p.setWeight(obs)
    
    def resample(self):
        #Choose particles to keep with probability = weight of particle
        # Create array of cumulative particle weight sums
        # create a new particle array, start as empty
        sum = 0
        cum_sum: list[float] = [sum]
        new_particles: list[Particle] = []
        for i in range(len(self.particles)):
            sum += self.particles[i].weight
            cum_sum.append(sum)
        
        # randomly choose a number between 0-100
        # Add the particle whose cumulaive sum is greater than chosen number but whose prior particle's sum is less than the chosen number
        # repeat N times - N is the number of particles you started with
        while(len(new_particles) < len(self.particles)):
            randNum:float =  float(random.randrange(0, 1000, 1)) / 1000.0
            for i in range(1, len(cum_sum)):
                if(randNum <= cum_sum[i]) and (randNum > cum_sum[i-1]):
                    new_particles.append(self.particles(i-1))
            
        # set the new resampled particle array as the new self.particles
        if(len(new_particles) != len(self.particles)): 
            raise RuntimeError("new particle array must be same length as old particle array")
        else:
            self.particles = new_particles[:]
