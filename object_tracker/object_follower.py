
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
import time


class ObjectFollower(Node):

    def __init__(self):
        super().__init__('object_follower')

        self.image_sub = self.create_subscription(Point, "/detected_object", self.receive_point_callback, 10)
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel_tracker", 10)

        # Declaring parameters:
        self.declare_parameter("rcv_timeout_secs", 2.0)
        self.declare_parameter("forward_chase_speed", 0.1)
        self.declare_parameter("max_size_thresh", 0.1)
        self.declare_parameter("filter_value", 0.9)
        self.declare_parameter("kp", 1.0)
        self.declare_parameter("ki", 0.1)
        self.declare_parameter("kd", 0.25)
        
        # Retrieving parameters:
        self.rcv_timeout_secs = self.get_parameter('rcv_timeout_secs').get_parameter_value().double_value
        self.forward_chase_speed = self.get_parameter('forward_chase_speed').get_parameter_value().double_value
        self.max_size_thresh = self.get_parameter('max_size_thresh').get_parameter_value().double_value
        self.filter_value = self.get_parameter('filter_value').get_parameter_value().double_value
        self.kp = self.get_parameter('kp').get_parameter_value().double_value
        self.ki = self.get_parameter('ki').get_parameter_value().double_value
        self.kd = self.get_parameter('kd').get_parameter_value().double_value

        self.object_x = 0.0
        self.object_z = 0.0
        self.lastrcvtime = time.time() - 10000

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.calcDesTwist)

        # PID Loop Variables:
        self.kp_term = 0 # Proportional term
        self.ki_term = 0 # Integral term
        self.kd_term = 0 # Derivative term

        self.err = 0 # Current error
        self.err_prev = 0 # Previous error
        
        self.max_ang_vel = 100 # Max angular velocity

        self.PIDNotActive = True

        self.get_logger().info('Initiated node')



    def receive_point_callback(self, data):
        
        # Print the data.x, data.y, and data.z values using the logger
        # self.get_logger().info("x={}, y={}, z={}".format(data.x, data.y, data.z))
        # self.get_logger().info("x={}".format(data.x, data.y, data.z))


        # Employing simple low-pass filter to filter incoming velocity values:
        f = self.filter_value
        self.object_x = self.object_x * f + data.x * (1-f)
        self.object_z = self.object_z * f + data.z * (1-f)
        self.lastrcvtime = time.time()
        
        return


    def calcDesTwist(self):
        # Method to calculate the desired twist commands to allow the robot to follow the object of interest

        msg = Twist()
        if(time.time() - self.lastrcvtime < self.rcv_timeout_secs):
            
            # if(self.object_z < self.max_size_thresh):
            msg.linear.x = self.forward_chase_speed
        
            # If the ball is back in view after previously not being in view, then (re)initialize the PID loop
            if(self.PIDNotActive):
                self.Initialize()
                self.PIDNotActive = False

            msg.angular.z = self.calcPIDAngVel()
        else:
            self.PIDNotActive = True
            
        # self.get_logger().info('%f, %f' %(msg.linear.x, msg.angular.z))
        self.vel_pub.publish(msg)
        
    
    def calcPIDAngVel(self):
        # Error is simply equal to the negative of self.object_x as it's desired to drive this state to 0
        
        self.err = -self.object_x

        # Calculating integral term, accounting for integral windup:
        if(self.ki_term > self.max_ang_vel):
            self.ki_term = self.max_ang_vel
        elif(self.ki_term < -self.max_ang_vel):
            self.ki_term = -self.max_ang_vel

        self.kp_term = self.kp * self.err
        self.kd_term = self.kd * (10 * (self.err - self.err_prev))

        ang_vel = self.kp_term + self.ki_term + self.kd_term


        # Updating variables:
        self.err_prev = self.err
        self.ki_term = self.ki_term + (self.ki * self.err * 0.1)

        # Saturating angular velocity command, if necessary:
        ang_vel = self.satAngVel(ang_vel)



        return ang_vel


    def satAngVel(self, ang_vel):
        # Method to saturate the angular velocity commands

        if(ang_vel > self.max_ang_vel):
            return self.max_ang_vel
        elif(ang_vel < -self.max_ang_vel):
            return -self.max_ang_vel
        else:
            return ang_vel


    def Initialize(self):        
        # Method to (re)initialize the PID loop
        # This is accomplished by setting the integral term to zero, as well as setting the previous error to equal the most recently-calculated error 

        self.err_prev = self.err
        self.ki_term = 0

        return





def main(args=None):
    rclpy.init(args=args)
    object_follower = ObjectFollower()
    
    # rclpy.spin(object_follower)
    # object_follower.destroy_node()
    # rclpy.shutdown()

    while rclpy.ok():
        rclpy.spin_once(object_follower)
        
    object_follower.destroy_node()
    rclpy.shutdown()
        