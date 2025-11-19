import rclpy
from rclpy.node import Node
from .Particle import Particle
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from example_interfaces.msg import UInt8
from example_interfaces.msg import Float32
import yaml
from builtin_interfaces.msg import Time
import random
import math


class ParticleFilter(Node):
    def __init__(self):
        super().__init__('filter')
        self.map: OccupancyGrid = OccupancyGrid()
        self.particles: list[Particle] = []
        self.curr_angle:float = 0.0

        # publish map to /floor topic
        self.map_pub = self.create_publisher(OccupancyGrid, '/floor', 10)
        self.declare_parameter('world_file', '')
        self.pubMap() # Intially create and publish the map

        # populate particles evenly over map, with the same weight
        num_particles = 1000
        init_weight:float = 1/num_particles
        particles_per_col = num_particles/self.map.info.width
        col_spacing = self.map.info.height*self.map.info.resolution / particles_per_col

        for i in range(self.map.info.width):
            for j in range(particles_per_col):
                particle_pose = Pose2D(x=self.map.info.width * (i + 0.5), y=col_spacing*j, theta=self.curr_angle)
                map_row:int = int((col_spacing*j) / self.map.info.resolution)
                color = "light" if self.map.data[map_row][i] == '.' else "dark"
                new_particle = Particle(particle_pose, color, 0) # No observation for this particle, inserted place holder to create particle, then force set weight
                new_particle.weight = init_weight
                self.particles.append(new_particle)
        # publish map every 5 seconds
        self.map_timer = self.create_timer(5, self.pubMap)

        # Publish best guess pose to /estimated_pose topic
        self.est_pose = self.create_publisher(Pose2D, '/estimated_pose', 10)
        self.best_estimate = self.create_timer(2, self.pubBestPosition) #publish every 2 seconds

        #Subscribe to /cmd_vel topic
        self.acrtion_msgs = self.create_subscription(Twist, '/cmd_vel', self.getAction, 10)
        #Sub to /floor_sensor topic
        self.obs_msgs = self.create_subscription(UInt8,'/floor_sensor', self.getObservation, 10)
        # sub to /compass topic
        self.compass_msgs = self.create_subscription(Float32, '/compass', self.getAngle, 10)
        
        

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
        # TODO: What about clusters? Weighted average won't deal well with multiple clusters
        best_pose: Pose2D = Pose2D(x=0, y=0, theta=0)
        for p in self.particles:
            best_pose = Pose2D(x=best_pose.x + p.state.x * p.weight, y=best_pose.y + p.state.y * p.weight, theta=best_pose.theta + p.state.theta * p.weight)

        msg:Pose2D = best_pose

        self.est_pose.publish(msg)

    def getAngle(self, msg:Float32):
        self.curr_angle = msg.data

    def getObservation(self, msg:UInt8):
        newObs: int = msg.data
        #After getting the new observation, reweight each particle
        self.reweight(newObs)

        #After reweighting all particles, resample them
        self.resample()

    def getAction(self, msg:Twist):
        lin_vel = msg.linear.x
        ang_vel = msg.angular.z

        #After getting the new action, forward projection each particle
        self.forwardProjection(lin_vel, ang_vel)

    def forwardProjection(self, lin_vel, ang_vel):
        # forward project movement of particle based on action
        if(lin_vel != 0) and (ang_vel != 0):
            for p in self.particles:
                new_x = p.state.x + lin_vel/ang_vel * (math.sin(self.curr_angle) - math.sin(p.state.theta))
                new_y = p.state.y - lin_vel/ang_vel * (math.cos(self.curr_angle) - math.cos(p.state.theta))
                p.state = Pose2D(x=new_x, y=new_y, theta=self.curr_angle)
        elif lin_vel != 0:
            for p in self.particles:
                new_x = p.state.x + lin_vel * math.cos(p.state.theta)
                new_y = p.state.y + lin_vel * math.sin(p.state.theta)
                p.state = Pose2D(x=new_x, y=new_y, theta=self.curr_angle)
        elif ang_vel != 0:
            for p in self.particles:
                p.state.theta = self.curr_angle        
                

        # Simulate the noise in the movements
        std_dev = 0.1 # The spread on the gaussian noise TODO: Fine tune
        for p in self.particles:
            # Add gaussian noise to new position
            p.state.x += random.gauss(0, std_dev)
            p.state.y += random.gauss(0, std_dev)


            # Update expected color
            map_row:int = math.floor(p.state.y/self.map.info.resolution)
            map_col:int = math.floor(p.state.x/self.map.info.resolution)
            p.color = "light" if self.map.data[map_row][map_col] == '.' else "dark"
         

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
        sum = 0
        cum_sum: list[float] = [sum]
        new_particles: list[Particle] = []
        for i in range(len(self.particles)):
            sum += self.particles[i].weight
            cum_sum.append(sum)
        
        # Add the particle whose cumulaive sum is greater than chosen number but whose prior particle's sum is less than the chosen number
        while(len(new_particles) < len(self.particles)):
            randNum:float =  float(random.randrange(1, 1000, 1)) / 1000.0
            for i in range(1, len(cum_sum)):
                if(randNum <= cum_sum[i]) and (randNum > cum_sum[i-1]):
                    new_particles.append(self.particles(i-1))
            
        if(len(new_particles) != len(self.particles)): 
            raise RuntimeError("new particle array must be same length as old particle array")
        else:
            self.particles = new_particles[:]


def main():
    rclpy.init()

    filter = ParticleFilter()

    rclpy.spin(filter)

    filter.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
