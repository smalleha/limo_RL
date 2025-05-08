#!/usr/bin/python3
import rospy
import numpy as np
import math
import copy
import time
from math import pi
from .respawnGoal import Respawn
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class Env():
    def __init__(self,action_dim = 2):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.initGoal = False
        self.get_goalbox = False
        
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        # self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        # self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        # self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        
        self.last_distance = 0
        self.past_distance = 0.
        self.initial_diatance = 0.
        self.stopped = 0
        self.action_dim = action_dim

        # self.x_gap_last = 0
        # self.y_gap_last = 0
        
        #Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)
        
    def shutdown(self):
        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)
    
    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.past_distance = goal_distance
        self.initial_diatance = goal_distance
        return goal_distance
    
    def getOdometry(self, odom):
        # 获得机器人的具体位置
        self.past_position = copy.deepcopy(self.position)
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        heading = goal_angle - yaw
        
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 3)
        
    def getState(self, scan,past_action):
        #state 20个激光雷达数据 + heading + current_disctance + obstacle_min_range, obstacle_angle 
        scan_range = []
        heading = self.heading
        min_range = 0.20
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf') or scan.ranges[i] >3.5:
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
        
        # min state 
        obstacle_min_range = round(min(scan_range), 2)

        # obstacle_angle = np.argmin(scan_range)
        # x_gap  = self.goal_x - self.position.x
        # y_gap = self.goal_y - self.position.y
        
        if min_range > min(scan_range) > 0:
            print("scan_range",scan_range)
            print("min_range",min_range)
            print("min(scan_range)",min(scan_range))
            done = True

            
        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance < 0.3:
            self.get_goalbox = True

        # return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done
        return scan_range + [heading, current_distance, obstacle_min_range,past_action[0],past_action[1]], done
    
    def setReward(self, state, action,done):

        current_distance = state[-4]
        heading = state[-5]
        obstacle_min_range = state[-3]

        # 距离奖励 
        distance_reward = -current_distance
        
        # 方向奖励
        turn_reward = -abs(heading)

        # 躲避障碍物体 Reward
        if obstacle_min_range < 0.8:
            ob_reward = -2 ** (0.6/obstacle_min_range)
        else:
            ob_reward = 0
        reward = distance_reward + turn_reward + ob_reward

        if done:
            rospy.loginfo("Collision!!")
            reward = -200.
            self.pub_cmd_vel.publish(Twist())
            self.respawn_goal.index = 0

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1600.
            self.pub_cmd_vel.publish(Twist())
            done = True
            # self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            # self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
            

        return reward, done
     
    def step(self, action,past_action):
        linear_vel = action[0]
        ang_vel = action[1]
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('limo/scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data, past_action)     
        reward, done = self.setReward(state, action,done)
        
        return np.asarray(state), reward, done

    def set_goal(self):
        str_goalx = input("please enter goalx: ")
        str_goaly = input("please enter goaly: ")
        self.goal_x = float(str_goalx)
        self.goal_y = float(str_goaly)
        self.respawn_goal.getPosition(str_goalx,str_goaly,True,delete=True)

    
    def reset(self):
        # rospy.wait_for_service('gazebo/reset_world')
        # try:
        #     self.reset_proxy()
        # except (rospy.ServiceException) as e:
        #     print("gazebo/reset_simulation service call failed")

        data = None
        
        while data is None:
            try:
                data = rospy.wait_for_message('/limo/scan', LaserScan, timeout=5)
                self.set_goal()
            except:
                pass
        
        # if self.initGoal:
        #     self.goal_x, self.goal_y = self.respawn_goal.getPosition()
        #     self.initGoal = False
            
        ## mabe debug
        # else:
        #     self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
           

        print("reset successfully")
        self.goal_distance = self.getGoalDistace()
        state, _ = self.getState(data, [0]*self.action_dim)
        return np.asarray(state)
    

        