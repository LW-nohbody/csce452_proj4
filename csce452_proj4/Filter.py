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
import numpy as np
from sklearn.cluster import DBSCAN


# TWIST_MSG_PERIOD = 0.25
# TWIST_TURN_MSG_PERIOD = 0.6
PARTICLE_SPACING = 0.1
PARTICLE_NUMBER = 1000
DEBUG = False
TESTING = False
# TESTING_X = 2.21
# TESTING_Y = 5.64
# TESTING_THETA = -0.7
# TESTING_X = 6.28
# TESTING_Y = 3.13
# TESTING_THETA = -0.45
# TESTING_X = 9.80
# TESTING_Y = 6.52
# TESTING_THETA = 1.67
# TESTING_COLOR = "dark"
CLUSTER_EPS = 0.3
CLUSTER_SAMPLES = 30

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
        particle_resolution:int = self.map.info.resolution // PARTICLE_SPACING
        num_particles = self.map.info.width * self.map.info.height * particle_resolution
        if(num_particles > PARTICLE_NUMBER):
            # intialize with particle number
            particles_per_row_col = math.floor(math.sqrt(PARTICLE_NUMBER))
            width_spacing = self.map.info.width * self.map.info.resolution / (particles_per_row_col- 1)
            height_spacing = self.map.info.height * self.map.info.resolution / (particles_per_row_col- 1)
            init_weight:float = 1/PARTICLE_NUMBER

            for i in range(particles_per_row_col): # loops through the row
                for j in range(particles_per_row_col): # adds to the columns
                    init_theta = random.uniform(-math.pi, math.pi)
                    particle_pose = Pose2D(x=i*width_spacing, y=height_spacing*j, theta=init_theta)
                    map_row:int = (height_spacing*j) // self.map.info.resolution
                    map_col:int = (width_spacing * i) // self.map.info.resolution
                    if(map_row == self.map.info.height): map_row -= 1
                    if(map_col == self.map.info.width): map_col -= 1
                    map_index:int = int(self.map.info.width * map_row + map_col) # map.data is in row major order
                    if(map_index >= len(self.map.data)):
                        print(f"row: {map_row}, column: {map_col}, max row: {self.map.info.height}, max col: {self.map.info.width}")
                    color = "light" if self.map.data[map_index] == 0 else "dark"
                    new_particle = Particle(particle_pose, color, 0) # No observation for this particle, inserted place holder to create particle, then force set weight
                    new_particle.weight = init_weight
                    self.particles.append(new_particle)
        else:
            # Initialize with particle spacing
            particle_resolution:int = self.map.info.resolution // PARTICLE_SPACING
            num_particles = self.map.info.width * self.map.info.height * particle_resolution
            init_weight:float = 1/num_particles
            particles_in_row:int = int(self.map.info.width * particle_resolution)
            particles_in_col:int = int(self.map.info.height * particle_resolution)
            for i in range(particles_in_row):
                for j in range(particles_in_col):
                    init_theta = random.uniform(-math.pi, math.pi)
                    particle_pose = Pose2D(x=i*PARTICLE_SPACING, y=j*PARTICLE_SPACING, theta=init_theta)
                    map_row:int = j//particle_resolution
                    map_col:int = i//particle_resolution
                    map_index:int = int(map_row * self.map.info.width + map_col)
                    color = "light" if self.map.data[map_index] == 0 else "dark"

                    new_particle = Particle(particle_pose, color, 0) # No observation for this particle, inserted place holder to create particle, then force set weight
                    new_particle.weight = init_weight
                    self.particles.append(new_particle)

        
        # particle_resolution:int = self.map.info.resolution // PARTICLE_SPACING
        # num_particles = self.map.info.width * self.map.info.height * particle_resolution
        # init_weight:float = 1/num_particles
        # particles_in_row:int = int(self.map.info.width * particle_resolution)
        # particles_in_col:int = int(self.map.info.height * particle_resolution)
        # for i in range(particles_in_row):
        #     for j in range(particles_in_col):
        #         particle_pose = Pose2D(x=i*PARTICLE_SPACING, y=j*PARTICLE_SPACING, theta = self.curr_angle)
        #         map_row:int = j//particle_resolution
        #         map_col:int = i//particle_resolution
        #         map_index:int = int(map_row * self.map.info.width + map_col)
        #         color = "light" if self.map.data[map_index] == 0 else "dark"

        #         new_particle = Particle(particle_pose, color, 0) # No observation for this particle, inserted place holder to create particle, then force set weight
        #         new_particle.weight = init_weight
        #         self.particles.append(new_particle)
                
        
        if(TESTING):
            self.testing_particle = Particle(Pose2D(x=TESTING_X, y=TESTING_Y, theta=TESTING_THETA), TESTING_COLOR, 0)
            self.test_pub = self.create_publisher(Pose2D, '/true_pos', 10)
        # publish map every 5 seconds
        self.map_timer = self.create_timer(5, self.pubMap)

        # Publish best guess pose to /estimated_pose topic
        self.est_pose = self.create_publisher(Pose2D, '/estimated_pose', 10)
        self.best_estimate = self.create_timer(2, self.pubBestPosition) #publish every 2 seconds


        # save last Twist msg and time received
        # self.last_twist_time = 0.0
        self.last_twist_msg: Twist = Twist()
        # self.last_twist_angle = 0.0
        self.last_update_time = self.get_clock().now()
        #Subscribe to /cmd_vel topic
        self.acrtion_msgs = self.create_subscription(Twist, '/cmd_vel', self.getAction, 10)
        self.motion_timer = self.create_timer(0.1, self.motion_update_loop)
        #Sub to /floor_sensor topic
        self.new_obs: int = 0
        self.obs_msgs = self.create_subscription(UInt8,'/floor_sensor', self.getObservation, 10)
        # sub to /compass topic
        self.compass_msgs = self.create_subscription(Float32, '/compass', self.getAngle, 10)
        
        

    def pubMap(self):
        value = self.get_parameter('world_file').get_parameter_value().string_value

        with open(value, 'r') as f:
            map_yaml = yaml.safe_load(f)
        
        reso = map_yaml["resolution"]
        lines = map_yaml["map"].splitlines()

        lines = [line for line in lines if line.strip()]

        width = len(lines[0])
        height = len(lines)

        #Format as occupancy grid and publish
        msg = OccupancyGrid()
        msg.header.frame_id = 'world'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.info.resolution = reso
        msg.info.width = width
        msg.info.height = height
        msg.info.origin = Pose() #Defaults to origin

        map_data = []
        for row in reversed(lines):
            for char in row:
                if char == '.':
                    map_data.append(0) #light
                elif char == '#':
                    map_data.append(100) # dark
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

        sum_sin_theta = 0.0
        sum_cos_theta = 0.0

        #Get clusters
        temp = []
        for p in self.particles:
            temp.append([p.state.x, p.state.y])
        
        np_temp = np.array(temp)
        db = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_SAMPLES)
        fitted_list = db.fit(np_temp)
        group_sizes = {}

        # Find sizes of each cluster
        for label in fitted_list.labels_:
            if label == -1: continue
            if(label in group_sizes):
                group_sizes[label] += 1
            else:
                group_sizes[label] = 1
        
        # Find largest cluster
        max_group_label = -1
        max_group_size = 0
        for label in group_sizes:
            if(group_sizes[label] > max_group_size):
                max_group_size = group_sizes[label]
                max_group_label = label
        
        cluster_particles:list[Particle] = []
        for i in range(len(fitted_list.labels_)):
            if(fitted_list.labels_[i] == max_group_label):
                cluster_particles.append(self.particles[i])

        # Average cluster weights
        for p in cluster_particles:
            if p.color == "invalid":
                continue
            valid_particles += 1
            weight = p.weight
            best_pose.x += p.state.x * weight
            best_pose.y += p.state.y * weight
            # best_pose.theta += p.state.theta * weight
            sum_weight += weight
            sum_sin_theta += math.sin(p.state.theta) * weight
            sum_cos_theta += math.cos(p.state.theta) * weight
        # for p in self.particles:
        #     if p.color == "invalid":
        #         continue
        #     valid_particles += 1
        #     weight = p.weight
        #     best_pose.x += p.state.x * weight
        #     best_pose.y += p.state.y * weight
        #     best_pose.theta += p.state.theta * weight
        #     sum_weight += weight
        #     sum_sin_theta += math.sin(p.state.theta) * weight
        #     sum_cos_theta += math.cos(p.state.theta) * weight
        
        if sum_weight > 0.0:
            best_pose.x /= sum_weight
            best_pose.y /= sum_weight

            avg_sin_theta = sum_sin_theta / sum_weight
            avg_cos_theta = sum_cos_theta / sum_weight
            avg_theta = math.atan2(avg_sin_theta, avg_cos_theta)
            best_pose.theta = avg_theta
            # best_pose.theta /= sum_weight
            self.get_logger().info(f"Published estimated pose: x={best_pose.x:.3f}, y={best_pose.y:.3f}, theta={best_pose.theta:.3f} (valid particles: {valid_particles}/{len(self.particles)})")
        else:
            self.get_logger().warn(f"total weight 0 - cannot publish pose. Valid particles: {valid_particles}/{len(self.particles)}")
            return

        msg:Pose2D = best_pose
        self.est_pose.publish(msg)
        if(TESTING):
            self.test_pub.publish(self.testing_particle.state)
    
    def motion_update_loop(self):
        now = self.get_clock().now()
        duration = (now - self.last_update_time).nanoseconds * (1e-9)
        self.last_update_time = now
        MAX_DURATION = 0.5
        if duration > MAX_DURATION:
            self.get_logger().warn(f"Duration since last motion update too high ({duration:.3f}s), capping to {MAX_DURATION}s")
            duration = MAX_DURATION

        last_twist_angle = self.curr_angle
        if duration > 0:
            self.forwardProjection(self.last_twist_msg.linear.x, self.last_twist_msg.angular.z, duration, last_twist_angle)

    def getAngle(self, msg:Float32):
        self.curr_angle = msg.data

    def getObservation(self, msg:UInt8):
        self.new_obs = msg.data
        #After getting the new observation, reweight each particle
        self.reweight(self.new_obs)

        # #After reweighting all particles, resample them
        self.resample()

    def getAction(self, msg:Twist):
        # if(self.last_twist_msg == None):
        #     self.last_twist_msg = msg
        #     self.last_twist_time = self.get_clock().now()
        #     self.last_twist_angle = self.curr_angle
        # else:
        #     #After getting the new action, forward projection each particle based on last action
        #     time_received = self.get_clock().now()
        #     duration = (time_received - self.last_twist_time).nanoseconds * (1e-9)
        #     self.forwardProjection(self.last_twist_msg.linear.x, self.last_twist_msg.angular.z, duration, self.last_twist_angle)
        #     self.last_twist_msg = msg
        #     self.last_twist_time = time_received
        #     self.last_twist_angle = self.curr_angle
        self.last_twist_msg = msg
        

    def forwardProjection(self, lin_vel, ang_vel, twist_time, last_twist_angle):
        # forward project movement of particle based on action
        if(lin_vel != 0) and (ang_vel != 0):
            if(TESTING):
                old_theta = last_twist_angle
                radius = lin_vel / ang_vel

                # center_x = self.testing_particle.state.x - radius * math.sin(old_theta)
                # center_y = self.testing_particle.state.y + radius * math.cos(old_theta)

                distance = lin_vel * twist_time
                new_x = self.testing_particle.state.x + distance * math.cos(last_twist_angle)
                new_y = self.testing_particle.state.y + distance * math.sin(last_twist_angle)
                self.testing_particle.state = Pose2D(x=new_x, y=new_y, theta=self.curr_angle)

                # new_theta = last_twist_angle
                # new_x = center_x + radius * math.sin(new_theta)
                # new_y = center_y - radius * math.cos(new_theta)
                # self.testing_particle.state = Pose2D(x=new_x, y=new_y, theta=new_theta)
            for p in self.particles:
                # old_theta = last_twist_angle
                # radius = lin_vel / ang_vel

                # center_x = p.state.x - radius * math.sin(old_theta)
                # center_y = p.state.y + radius * math.cos(old_theta)

                distance = lin_vel * twist_time
                new_x = p.state.x + distance * math.cos(last_twist_angle)
                new_y = p.state.y + distance * math.sin(last_twist_angle)
                new_theta = self.curr_angle + random.gauss(0, 0.05)
                p.state = Pose2D(x=new_x, y=new_y, theta=new_theta)

                # new_theta = self.curr_angle + random.gauss(0, 0.05)
                # new_x = center_x + radius * math.sin(new_theta)
                # new_y = center_y - radius * math.cos(new_theta)
                # p.state = Pose2D(x=new_x, y=new_y, theta=new_theta)
        elif lin_vel != 0:
            if(TESTING):
                new_x = self.testing_particle.state.x + lin_vel * math.cos(last_twist_angle) * twist_time
                new_y = self.testing_particle.state.y + lin_vel * math.sin(last_twist_angle) * twist_time
                self.testing_particle.state = Pose2D(x=new_x, y=new_y, theta=self.curr_angle)
            for p in self.particles:
                # p.state.theta = self.curr_angle
                new_x = p.state.x + lin_vel * math.cos(last_twist_angle) * twist_time
                new_y = p.state.y + lin_vel * math.sin(last_twist_angle) * twist_time
                p.state = Pose2D(x=new_x, y=new_y, theta=self.curr_angle + random.gauss(0, 0.05))
        elif ang_vel != 0:
            if(TESTING):
                self.testing_particle.state.theta = last_twist_angle
            for p in self.particles:
                # p.state.theta = self.curr_angle + ang_vel * twist_time  
                p.state.theta = self.curr_angle + random.gauss(0, 0.05)
      

        # Simulate the noise in the movements
        std_dev = 0.1 # The spread on the gaussian noise TODO: Fine tune
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
        if(num_out_of_bounds == len(self.particles)):
                # raise RuntimeError(f"All particles invalid from forward projection, lin_vel: {lin_vel}, ang_vel: {ang_vel}, time: {twist_time}")
                self.get_logger().warn(f"All particles out of bounds from forward projection, lin_vel: {lin_vel}, ang_vel: {ang_vel}, time: {twist_time}")
                self.reinitializeParticles()
                return
        
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
            self.get_logger().info(f"After projection: x in [{minx:.2f}, {maxx:.2f}], y in [{miny:.2f}, {maxy:.2f}], time: {twist_time}")
        
        # After projecting, reweight with current observation
        # self.reweight(self.new_obs)
         

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
        
        # After reweighting, resample particles
        self.resample()
    
    # def resample(self):
    #     #Choose particles to keep with probability = weight of particle
    #     sum = 0
    #     cum_sum: list[float] = [sum]
    #     new_particles: list[Particle] = []
    #     for i in range(len(self.particles)):
    #         sum += self.particles[i].weight
    #         cum_sum.append(sum)
        
    #     # self.get_logger().info("Entering resampling loop")

    #     if sum == 0.0:
    #         self.get_logger().warn("All weights are 0 - reinitializing particles across map")
    #         # Reinitialize particles when all weights are 0 (recovery mechanism)
    #         self.reinitializeParticles()
    #         return
        
    #     # Add the particle whose cumulaive sum is greater than chosen number but whose prior particle's sum is less than the chosen number
        
    #     while(len(new_particles) < len(self.particles)):
    #         randNum:float = random.uniform(0, sum)
    #         found = False

    #         for i in range(1, len(cum_sum)):
    #             if(randNum <= cum_sum[i]) and (randNum > cum_sum[i-1]):
    #                 # Create a copy of the particle instead of using the same reference
    #                 new_particles.append(self.particles[i-1].copy())
    #                 found = True
    #                 break
    #         if(DEBUG and not found):
    #             print(f"Cant find weight for random number: {randNum}")
                    
    #     if(len(new_particles) != len(self.particles)) or (new_particles == []): 
    #         self.get_logger().info("ERROR: Particle arrays differ")
    #         raise RuntimeError("new particle array must be same length as old particle array")
    #     else:
    #         self.particles = new_particles
    
    def resample(self):
        new_particles: list[Particle] = []
        num_particles = len(self.particles)

        sum = 0

        for p in self.particles:
            sum += p.weight
        
        if sum == 0.0:
            self.get_logger().warn("All weights are 0 - reinitializing particles across map")
            self.reinitializeParticles()
            return
        
        if abs(sum-1.0) > 1e-6:
            for p in self.particles:
                p.weight /= sum
        
        interval = 1.0 / num_particles
        rand_start = random.uniform(0, interval)
        cum_sum = 0.0
        index = 0
        
        for i in range(num_particles):
            target_weight = rand_start + (i * interval)
            while target_weight > cum_sum:
                cum_sum += self.particles[index].weight
                index += 1
            
            new_particles.append(self.particles[index - 1].copy())
        
        if len(new_particles) != num_particles:
            self.get_logger().error(f"Error in resampling, expected {num_particles} but actual length {len(new_particles)}")
            return
        
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
                init_theta = random.uniform(-math.pi, math.pi)
                particle_pose = Pose2D(x=self.map.info.resolution * (i + 0.5), y=col_spacing*j, theta=init_theta)
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
