#!/usr/bin/env python
#################################################################################
# 版权所有 © 2018 ROBOTIS CO., LTD.
# 本代码使用 Apache 2.0 协议开源。
#################################################################################

# 作者: Gilbert #

# 导入需要用到的模块
import rospy  # ROS Python接口库
import random  # 用于生成随机数
import time  # 时间模块
import os  # 与文件路径相关操作
import math  # 数学计算模块
from gazebo_msgs.srv import SpawnModel, DeleteModel  # Gazebo服务：模型生成与删除
from gazebo_msgs.msg import ModelStates  # Gazebo话题：获取世界中所有模型的状态
from geometry_msgs.msg import Pose, PoseStamped  # 位置相关的消息类型

# 定义 Respawn 类用于目标点的管理
class Respawn():
    def __init__(self):
        # 获取当前脚本的绝对路径
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        # 替换路径，将脚本路径替换为目标模型文件路径
        self.modelPath = self.modelPath.replace('scripts/DDPG',
                                                'models/turtlebot3_square/goal_box/model.sdf')
        # 打开模型文件并读取内容（SDF模型）
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()

        # 获取ROS参数服务器上的阶段编号（stage_number）
        self.stage = rospy.get_param('/stage_number')

        # 初始化目标点的位置（Pose对象）
        self.goal_position = Pose()
        # 初始化用于接收RViz目标点的变量（PoseStamped对象）
        self.nav_goal = PoseStamped()
        
        # 设置默认目标点位置
        self.init_goal_x = 0
        self.init_goal_y = 0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y

        # 模型在 Gazebo 中的名称
        self.modelName = 'goal'
        
        # 设置障碍物（未使用）
        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6        

        # 用于记录上一次的目标点位置，避免生成太近
        # self.last_goal_x = self.init_goal_x
        # self.last_goal_y = self.init_goal_y
        self.last_goal_x = self.nav_goal.pose.position.x
        self.last_goal_y = self.nav_goal.pose.position.y

        # 上一次选取的索引值（避免重复）
        self.last_index = 0

        # 订阅 Gazebo 中模型状态话题，用于检测模型是否存在
        # self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False  # 是否检测到目标模型
        self.index = 0  # 当前目标索引
        self.index_num = 0  # 暂未使用
        self.R_num = 1  # 暂未使用

        # 订阅来自 RViz 的 2D Nav Goal，用于设置目标点
        self.nav_goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.navGoalCallback)

    # RViz中点击目标点后触发的回调函数
    def navGoalCallback(self, msg):
        # 将 RViz 传来的目标点保存到 nav_goal 中
        self.nav_goal.pose.position.x = msg.pose.position.x 
        self.nav_goal.pose.position.y = msg.pose.position.y 
        rospy.loginfo("Received new 2D Nav Goal from RViz")

    # Gazebo模型状态话题的回调，用于检测目标模型是否存在
    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True  # 如果检测到名为"goal"的模型，则标记为已存在

    # 生成目标点模型
    def respawnModel(self,goal_x,goal_y):
        while True:
            if not self.check_model:
                print("111")  # debug输出
                # 等待 Gazebo 模型生成服务可用
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                # 创建服务代理
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                # 发送生成模型请求
                self.goal_position.x = goal_x
                self.goal_position.y = goal_y
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", goal_x,
                              goal_y)
                break
            else:
                print("222")  # debug输出
                pass

    # 删除目标点模型
    def deleteModel(self):
        while True:
            if self.check_model:
                # 等待服务可用
                rospy.wait_for_service('gazebo/delete_model')
                # 创建服务代理
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                # 请求删除模型
                del_model_prox(self.modelName)
                break
            else:
                pass

    # 获取目标点位置，可以选择删除当前模型并生成新的
    def getPosition(self, x, y, position_check=False, delete=False):
        # 如果需要删除现有模型
        # if delete:
        #     self.deleteModel()

        # 如果是stage 4 以外的情况，使用随机生成目标点
        if self.stage != 4:
            # 限定目标点生成的坐标范围
            x_min, x_max = -2.5, 2.5
            y_min, y_max = -2.5, 2.5
            min_distance = 0.5  # 与上一个目标的最小距离

            while position_check:
                # 随机生成坐标
                # x = round(random.uniform(x_min, x_max), 2)
                # y = round(random.uniform(y_min, y_max), 2)

                # x = self.nav_goal.pose.position.x
                # y = self.nav_goal.pose.position.y

                # 计算新坐标与上一个目标点的距离
                dist = math.hypot(x - self.last_goal_x, y - self.last_goal_y)
                if dist < min_distance:
                    continue  # 距离太近，重试

                # 设置为新的目标点
                self.goal_position.position.x = x
                self.goal_position.position.y = y
                position_check = False  # 退出循环

        else:
            # stage == 4 时，使用固定的一组点生成目标
            while position_check:
                # 使用正方形加对角的8个点
                R = 0.6 * math.sin(math.pi / 2)
                goal_x_list = [0.6, -0.6, 0, 0, R, R, -R, -R]
                goal_y_list = [0, 0, 0.6, -0.6, R, -R, R, -R]
                
                # 随机选择一个目标点索引
                self.index = random.randrange(0, 8)
                print(self.index, self.last_index)
                if self.last_index == self.index:
                    position_check = True  # 如果跟上次相同则继续循环
                else:
                    self.last_index = self.index
                    position_check = False  # 可接受，退出循环

                # 设置目标点坐标
                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]

        # 等待0.5秒，避免过快
        time.sleep(0.5)

        # 生成目标模型
        # self.respawnModel()

        # 保存当前目标坐标，用于下次对比距离
        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        # 返回目标坐标
        return self.goal_position.position.x, self.goal_position.position.y
