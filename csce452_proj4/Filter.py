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


TWIST_MSG_PERIOD = 0.25
DEBUG = False
TESTING = True
TESTING_X = 2.21
TESTING_Y = 5.64
TESTING_THETA = -0.7
TESTING_COLOR = "light"

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
        particles_per_col:int = math.ceil(num_particles/self.map.info.width)
        col_spacing:float = self.map.info.height*self.map.info.resolution / particles_per_col

        for i in range(self.map.info.width): # loops through the row
            for j in range(particles_per_col): # adds to the columns
                particle_pose = Pose2D(x=self.map.info.resolution * (i + 0.5), y=col_spacing*j, theta=self.curr_angle)
                map_row:int = int((col_spacing*j) / self.map.info.resolution)
                map_index:int = self.map.info.width * map_row + i # map.data is in row major order
                color = "light" if self.map.data[map_index] == 0 else "dark"
                new_particle = Particle(particle_pose, color, 0) # No observation for this particle, inserted place holder to create particle, then force set weight
                new_particle.weight = init_weight
                self.particles.append(new_particle)
        
        if(TESTING):
            self.testing_particle = Particle(Pose2D(x=TESTING_X, y=TESTING_Y, theta=TESTING_THETA), TESTING_COLOR, 0)
            self.test_pub = self.create_publisher(Pose2D, '/true_pos', 10)
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
        sum_weight = 0.0
        valid_particles = 0

        for p in self.particles:
            if p.color == "invalid":
                continue
            valid_particles += 1
            best_pose.x += p.state.x * p.weight
            best_pose.y += p.state.y * p.weight
            best_pose.theta += p.state.theta * p.weight
            sum_weight += p.weight
        
        if sum_weight > 0.0:
            best_pose.x /= sum_weight
            best_pose.y /= sum_weight
            best_pose.theta /= sum_weight
            self.get_logger().info(f"Published estimated pose: x={best_pose.x:.3f}, y={best_pose.y:.3f}, theta={best_pose.theta:.3f} (valid particles: {valid_particles}/{len(self.particles)})")
        else:
            self.get_logger().warn(f"total weight 0 - cannot publish pose. Valid particles: {valid_particles}/{len(self.particles)}")
            return

        msg:Pose2D = best_pose
        self.est_pose.publish(msg)
        if(TESTING):
            self.test_pub.publish(self.testing_particle.state)

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
            if(TESTING):
                new_theta = self.testing_particle.state.theta + ang_vel * TWIST_MSG_PERIOD
                new_x = self.testing_particle.state.x + lin_vel/ang_vel * (math.sin(new_theta) - math.sin(self.testing_particle.state.theta)) * TWIST_MSG_PERIOD
                new_y = self.testing_particle.state.y - lin_vel/ang_vel * (math.cos(new_theta) - math.cos(self.testing_particle.state.theta)) * TWIST_MSG_PERIOD
                self.testing_particle.state = Pose2D(x=new_x, y=new_y, theta=new_theta)
            for p in self.particles:
                new_theta = p.state.theta + ang_vel * TWIST_MSG_PERIOD
                new_x = p.state.x + lin_vel/ang_vel * (math.sin(new_theta) - math.sin(p.state.theta)) * TWIST_MSG_PERIOD
                new_y = p.state.y - lin_vel/ang_vel * (math.cos(new_theta) - math.cos(p.state.theta)) * TWIST_MSG_PERIOD
                p.state = Pose2D(x=new_x, y=new_y, theta=new_theta)
        elif lin_vel != 0:
            if(TESTING):
                new_x = self.testing_particle.state.x + lin_vel * math.cos(self.testing_particle.state.theta) * TWIST_MSG_PERIOD
                new_y = self.testing_particle.state.y + lin_vel * math.sin(self.testing_particle.state.theta) * TWIST_MSG_PERIOD
                self.testing_particle.state = Pose2D(x=new_x, y=new_y, theta=self.curr_angle)
            for p in self.particles:
                new_x = p.state.x + lin_vel * math.cos(p.state.theta) * TWIST_MSG_PERIOD
                new_y = p.state.y + lin_vel * math.sin(p.state.theta) * TWIST_MSG_PERIOD
                p.state = Pose2D(x=new_x, y=new_y, theta=self.curr_angle)
        elif ang_vel != 0:
            if(TESTING):
                self.testing_particle.state.theta = self.testing_particle.state.theta + ang_vel * TWIST_MSG_PERIOD
            for p in self.particles:
                p.state.theta = p.state.theta + ang_vel * TWIST_MSG_PERIOD        

        # Simulate the noise in the movements
        std_dev = 0.01 # The spread on the gaussian noise TODO: Fine tune
        num_out_of_bounds = 0
        for p in self.particles:
            # Add gaussian noise to new position
            p.state.x += random.gauss(0, std_dev)
            p.state.y += random.gauss(0, std_dev)

            if((p.state.x > (self.map.info.width * self.map.info.resolution))
                or (p.state.x < 0)
                or (p.state.y > (self.map.info.height * self.map.info.resolution)) 
                or (p.state.y < 0)
            ): num_out_of_bounds += 1

            if(num_out_of_bounds == len(self.particles)):
                raise RuntimeError(f"All particles invalid from forward projection, lin_vel: {lin_vel}, ang_vel: {ang_vel}")


            # Update expected color
            map_row:int = math.floor(p.state.y/self.map.info.resolution)
            map_col:int = math.floor(p.state.x/self.map.info.resolution)
            if(map_row < 0 or map_col < 0 or 
               map_row >= self.map.info.height or map_col >= self.map.info.width):
                # self.get_logger().info(f"Particle positioned at ({p.state.x}, {p.state.y}) is outside of map with width {self.map.info.width}, height {self.map.info.height}, and reso {self.map.info.resolution}")
                p.color = "invalid"
                continue
            map_index = map_col + (map_row * self.map.info.width)
            if map_index < 0 or map_index >= len(self.map.data):
                p.color = "invalid"
                continue
            p.color = "light" if self.map.data[map_index] == 0 else "dark"

        if(TESTING):
            map_row:int = math.floor(self.testing_particle.state.y/self.map.info.resolution)
            map_col:int = math.floor(self.testing_particle.state.x/self.map.info.resolution)
            map_index = map_col + (map_row * self.map.info.width)
            self.testing_particle.color = "light" if self.map.data[map_index] == 0 else "dark"
        
        if(DEBUG):
            minx = min(p.state.x for p in self.particles)
            maxx = max(p.state.x for p in self.particles)
            miny = min(p.state.y for p in self.particles)
            maxy = max(p.state.y for p in self.particles)
            self.get_logger().info(f"After projection: x in [{minx:.2f}, {maxx:.2f}], y in [{miny:.2f}, {maxy:.2f}]")
         

    def reweight(self, obs: int):
        for p in self.particles:
            # if particle is outside map or invalid, force weight to be 0
            if((p.state.x > (self.map.info.width * self.map.info.resolution))
                or (p.state.x < 0)
                or (p.state.y > (self.map.info.height * self.map.info.resolution)) 
                or (p.state.y < 0)
                or (p.color == "invalid")
            ):
                p.weight = 0
            else:
                p.setWeight(obs)
        sum_weights = sum(p.weight for p in self.particles)
        if sum_weights > 0:
            for p in self.particles:
                p.weight /= sum_weights
        else:
            self.get_logger().warn("All weights are zero after reweighting.")
    
    def resample(self):
        #Choose particles to keep with probability = weight of particle
        sum = 0
        cum_sum: list[float] = [sum]
        num_zero_sums = 1
        new_particles: list[Particle] = []
        for i in range(len(self.particles)):
            sum += self.particles[i].weight
            cum_sum.append(sum)
        
        self.get_logger().info("Entering resampling loop")

        if sum == 0.0:
            self.get_logger().warn("All weights are 0 - reinitializing particles across map")
            # Reinitialize particles when all weights are 0 (recovery mechanism)
            self.reinitializeParticles()
            return
        
        # Add the particle whose cumulaive sum is greater than chosen number but whose prior particle's sum is less than the chosen number
        
        while(len(new_particles) < len(self.particles)):
            randNum:float = random.uniform(0, sum)  # Fixed: should start at 0, not 1
            found = False

            for i in range(1, len(cum_sum)):
                if(randNum <= cum_sum[i]) and (randNum > cum_sum[i-1]):
                    # Create a copy of the particle instead of using the same reference
                    new_particles.append(self.particles[i-1].copy())
                    found = True
                    break
            if not found:
                self.get_logger().info(f"No particle found for randNum={randNum}, sum={sum}")
                # Create a copy of the last particle
                new_particles.append(self.particles[-1].copy())
            
            # print(len(new_particles))
        
        self.get_logger().info("exiting resample loop")
            
        if(len(new_particles) != len(self.particles)) or (new_particles == []): 
            self.get_logger().info("ERROR: Particle arrays differ")
            raise RuntimeError("new particle array must be same length as old particle array")
        else:
            self.particles = new_particles
    
    def reinitializeParticles(self):
        """Reinitialize particles evenly across the map when filter loses track"""
        num_particles = len(self.particles)
        init_weight:float = 1.0/num_particles
        particles_per_col:int = math.ceil(num_particles/self.map.info.width)
        col_spacing:float = self.map.info.height*self.map.info.resolution / particles_per_col
        
        self.particles = []
        for i in range(self.map.info.width):
            for j in range(particles_per_col):
                if len(self.particles) >= num_particles:
                    break
                particle_pose = Pose2D(x=self.map.info.resolution * (i + 0.5), y=col_spacing*j, theta=self.curr_angle)
                map_row:int = int((col_spacing*j) / self.map.info.resolution)
                map_index:int = self.map.info.width * map_row + i
                if map_index >= 0 and map_index < len(self.map.data):
                    color = "light" if self.map.data[map_index] == 0 else "dark"
                    new_particle = Particle(particle_pose, color, 0)
                    new_particle.weight = init_weight
                    self.particles.append(new_particle)
        
        # Fill remaining particles if needed
        while len(self.particles) < num_particles:
            x = random.uniform(0, self.map.info.width * self.map.info.resolution)
            y = random.uniform(0, self.map.info.height * self.map.info.resolution)
            particle_pose = Pose2D(x=x, y=y, theta=self.curr_angle)
            map_row = int(y / self.map.info.resolution)
            map_col = int(x / self.map.info.resolution)
            if map_row >= 0 and map_col >= 0 and map_row < self.map.info.height and map_col < self.map.info.width:
                map_index = map_col + (map_row * self.map.info.width)
                if map_index >= 0 and map_index < len(self.map.data):
                    color = "light" if self.map.data[map_index] == 0 else "dark"
                    new_particle = Particle(particle_pose, color, 0)
                    new_particle.weight = init_weight
                    self.particles.append(new_particle)


def main():
    rclpy.init()

    filter = ParticleFilter()

    rclpy.spin(filter)

    filter.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
