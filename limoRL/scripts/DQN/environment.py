#!/usr/bin/python3
import rospy  # ROS Python 接口库
import numpy as np  # 数学数组库
import math  # 提供数学运算函数
from math import pi  # π 圆周率常量
from geometry_msgs.msg import Twist, Point, Pose  # ROS 中的几何消息类型
from sensor_msgs.msg import LaserScan  # 激光雷达扫描数据类型
from nav_msgs.msg import Odometry  # 里程计数据类型
from std_srvs.srv import Empty  # 空服务，用于控制 Gazebo 重置等
from tf.transformations import euler_from_quaternion, quaternion_from_euler  # 欧拉角与四元数互转
from .respawnGoal import Respawn  # 自定义类，用于目标点管理


class Env():
    def __init__(self, action_size):
        # 初始化变量
        self.goal_x = 0  # 目标点X坐标
        self.goal_y = 0  # 目标点Y坐标
        self.heading = 0  # 当前朝向与目标点方向的夹角
        self.action_size = action_size  # 动作空间大小（离散动作数量）
        self.initGoal = True  # 是否首次生成目标点
        self.get_goalbox = False  # 是否到达目标点标志
        self.position = Pose()  # 当前位姿

        # ROS 发布与订阅设置
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)  # 发布速度指令
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)  # 订阅里程计信息
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)  # 重置仿真世界
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)  # 恢复仿真
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)  # 暂停仿真
        self.respawn_goal = Respawn()  # 初始化目标点生成器
        self.last_distance = 0  # 上一次与目标点的距离

    def getGoalDistace(self):
        # 计算与目标点之间的欧几里得距离
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        return goal_distance

    def getOdometry(self, odom):
        # 处理 odom 消息，提取位置与朝向
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]

        # 将四元数转换为欧拉角，得到偏航角 yaw
        _, _, yaw = euler_from_quaternion(orientation_list)

        # 计算目标方向与机器人朝向的夹角（heading）
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        heading = goal_angle - yaw

        # 将 heading 限制在 [-pi, pi]
        if heading > pi:
            heading -= 2 * pi
        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan, past_action):
        # 获取当前环境状态
        scan_range = []
        heading = self.heading
        min_range = 0.20  # 设置碰撞检测的最小距离阈值
        done = False

        for i in range(len(scan.ranges)):
            # 对激光数据进行清洗
            if scan.ranges[i] == float('Inf') or scan.ranges[i] > 3.5:
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)

        # 如果有障碍物过近，标记为 done（碰撞）
        if min_range > min(scan_range) > 0:
            done = True

        # 计算当前距离目标点的距离
        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        if current_distance < 0.3:
            self.get_goalbox = True

        # 状态包含：激光数据 + 朝向角 + 当前距离 + 最近障碍物距离 + 上一个动作
        return scan_range + [heading, current_distance, obstacle_min_range, past_action], done

    def setReward(self, state, done, action):
        # 根据状态计算奖励
        current_distance = state[-3]
        heading = state[-4]
        obstacle_min_range = state[-2]

        distance_reward = -current_distance  # 离目标越近，惩罚越小
        turn_reward = -abs(heading)  # 对准目标方向奖励更高

        if obstacle_min_range < 0.8:
            ob_reward = -2 ** (0.6 / obstacle_min_range)  # 障碍越近，惩罚越大
        else:
            ob_reward = 0

        reward = distance_reward + turn_reward + ob_reward

        if done:
            rospy.loginfo("Collision!!")
            reward = -200  # 碰撞严重惩罚
            self.pub_cmd_vel.publish(Twist())  # 停止机器人

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1600  # 成功到达目标点奖励
            self.pub_cmd_vel.publish(Twist())  # 停止机器人
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)  # 生成新目标点
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def Getang(self, past_action):
        # 将离散动作编号映射为角速度
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - past_action) * max_angular_vel * 0.5
        return ang_vel

    def step(self, action, past_action):
        # 执行动作，获取下一状态、奖励和是否终止

        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15  # 固定线速度
        vel_cmd.angular.z = ang_vel  # 设置角速度
        self.pub_cmd_vel.publish(vel_cmd)

        # 等待激光雷达数据
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/limo/scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data, past_action)
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self, past_action):
        # 重置仿真环境与目标点

        rospy.wait_for_service('gazebo/reset_world')
        try:
            self.reset_proxy()  # 调用 Gazebo 服务重置世界
        except rospy.ServiceException:
            print("gazebo/reset_simulation service call failed")

        print("self.respawn_goal.check_model1 :", self.respawn_goal.check_model)

        # 等待激光雷达准备好
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/limo/scan', LaserScan, timeout=5)
            except:
                pass

        print("self.respawn_goal.check_model2 :", self.respawn_goal.check_model)

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        print("reset successfully")

        self.goal_distance = self.getGoalDistace()
        self.last_distance = self.goal_distance

        state, done = self.getState(data, past_action)

        return np.asarray(state)
