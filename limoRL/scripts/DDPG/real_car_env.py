#!/usr/bin/env python3
"""
针对真实 Limo 小车的 RL 环境代码。
"""
import rospy
import numpy as np
import math
import random  # 用于生成随机数
from math import pi
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class Env:
    def __init__(self, action_size):
        """
        :param action_size: 离散动作数量
        """
        self.action_size = action_size
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.heading = 0.0
        self.initGoal = False
        self.get_goalbox = False
        self.position = Pose().position
        self.home_pose = (1.0, 0.0, 0.0)
        self.goal_position = Pose()

        # 发布速度指令
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        # 订阅里程计
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        # 订阅真实小车激光扫描
        self.sub_scan = rospy.Subscriber('/limo/scan', LaserScan, self._on_scan)
        self._latest_scan = None
        self.last_goal_x = 0.0  
        self.last_goal_y = 0.0

    def _on_scan(self, msg):
        """保存最新的激光数据"""
        self._latest_scan = msg

    def getOdometry(self, odom):
        """处理里程计，提取位置与朝向 yaw"""
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(quat)
        # 计算 heading
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2*pi
        elif heading < -pi:
            heading += 2*pi
        self.heading = heading

    def getGoalDistance(self):
        """计算当前与目标的欧氏距离"""
        return math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y)

    # def getState(self, scan, past_action):
    #     """
    #     生成环境状态：
    #       - 处理后的激光范围列表
    #       - 机器人朝向目标的角度 heading
    #       - 与目标的距离
    #       - 最近障碍物距离
    #       - 上一个动作编号
    #     返回 state(list) 和 done(bool) 碰撞标志
    #     """
    #     scan_range = []
    #     for r in scan.ranges:
    #         if np.isnan(r):
    #             scan_range.append(0.0)
    #         elif r == float('Inf') or r > 3.5:
    #             scan_range.append(3.5)
    #         else:
    #             scan_range.append(r)

    #     obstacle_min = round(min(scan_range), 2)
    #     done = obstacle_min < 0.2

    #     current_distance = round(self.getGoalDistance(), 2)
    #     if current_distance < 0.3:
    #         self.get_goalbox = True

    #     state = scan_range + [round(self.heading, 2), current_distance, obstacle_min, past_action]
    #     return state, done

    def getState(self, scan, past_action):
        """
        生成扁平状态列表和 done 标志：
          - scan_ranges (固定长度)
          - heading, distance, obstacle_min
          - past_action 向量各分量
        """
        # 1. 清洗激光数据
        scan_range = []
        for r in scan.ranges:
            if np.isnan(r):
                scan_range.append(0.0)
            elif r == float('Inf') or r > 3.5:
                scan_range.append(3.5)
            else:
                scan_range.append(r)

        # 2. 障碍检测
        obstacle_min = round(min(scan_range), 2)
        done = obstacle_min < 0.2

        # 3. 距离与目标判断
        distance = round(self.getGoalDistance(), 2)
        if distance < 0.3:
            self.get_goalbox = True

        # 4. 展平 past_action
        try:
            past_list = list(past_action)
        except Exception:
            past_list = [float(past_action)]

        # 5. 拼接成扁平列表
        state = []
        state.extend(scan_range)
        state.append(round(self.heading, 2))
        state.append(distance)
        state.append(obstacle_min)
        state.extend(past_list)

        return state, done

    def setReward(self, state, done, action):
        current_distance = state[-3]
        heading = state[-4]
        obstacle_min = state[-2]

        reward = -current_distance - abs(heading)
        if obstacle_min < 0.8:
            reward += -2**(0.6/obstacle_min)

        if done:
            rospy.logwarn("Collision! E-STOP")
            reward = -200
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal reached!")
            reward = 1600
            cmd = Twist()
            cmd.angular.z = 0
            cmd.linear.x = 0
            self.pub_cmd_vel.publish(cmd)
            
            self.getPosition()
            self.goal_distance = self.getGoalDistance()
            self.get_goalbox = False

        return reward

    def Getang(self, action):
        """离散动作映射到角速度"""
        max_ang = 1.5
        return ((self.action_size-1)/2 - action) * max_ang * 0.5

    def step(self, action, past_action):
        """执行动作，返回 next_state, reward, done"""
        # cmd = Twist()
        # cmd.linear.x = 0.15
        # cmd.angular.z = self.Getang(action)
        # self.pub_cmd_vel.publish(cmd)

        linear_vel = action[0]
        ang_vel = action[1]
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        while self._latest_scan is None and not rospy.is_shutdown():
            rospy.sleep(0.01)

        state, done = self.getState(self._latest_scan, past_action)
        reward = self.setReward(state, done, action)
        return np.asarray(state), reward, done

    def getPosition(self, position_check=True):

        # 限定目标点生成的坐标范围
        x_min, x_max = -2.5, 2.5
        y_min, y_max = -2.5, 2.5
        min_distance = 0.5  # 与上一个目标的最小距离

        while position_check:
            # 随机生成坐标
            x = round(random.uniform(x_min, x_max), 2)
            y = round(random.uniform(y_min, y_max), 2)

            # 计算新坐标与上一个目标点的距离
            dist = math.hypot(x - self.last_goal_x, y - self.last_goal_y)
            if dist < min_distance:
                continue  # 距离太近，重试

            # 设置为新的目标点
            self.goal_position.position.x = x
            self.goal_position.position.y = y
            print("x: ",x)
            print("y: ",y)
            position_check = False  # 退出循环

    def _go_home(self):
        """真实车：简单比例控制回到 home_pose"""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            dx = self.home_pose[0] - self.position.x
            dy = self.home_pose[1] - self.position.y
            dist = math.hypot(dx, dy)
            if dist < 0.05:
                break
            angle_to = math.atan2(dy, dx)
            cmd = Twist()
            cmd.linear.x = 0.1 * dist
            cmd.angular.z = 2.0 * angle_to
            self.pub_cmd_vel.publish(cmd)
            rate.sleep()

    def reset(self, past_action):
        """重置环境，返回初始 state"""
        # self.pub_cmd_vel.publish(Twist())
        
        if self.initGoal:
            self._go_home()
            self.initGoal = False
        else:
            self.getPosition()

        while self._latest_scan is None and not rospy.is_shutdown():
            rospy.sleep(0.01)

        state, done = self.getState(self._latest_scan, past_action)
        return np.asarray(state,dtype=float)


# if __name__ == '__main__':
#     rospy.init_node('rl_env_node')
#     env = Env(action_size=5)
#     past_action = 2
#     state = env.reset(past_action)
#     for t in range(1000):
#         action = np.random.randint(0, env.action_size)
#         state, reward, done = env.step(action, past_action)
#         past_action = action
#         rospy.loginfo(f"Step {t} - reward: {reward:.2f}, done: {done}")
#         if done:
#             state = env.reset(past_action)
#             past_action = 2
