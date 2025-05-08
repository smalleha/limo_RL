#!/usr/bin/env python

# 引入必要的库
import torch                                    # 导入 PyTorch 主库
import torch.nn as nn                           # 导入神经网络模块
import torch.nn.functional as F                 # 导入激活函数、损失函数等功能模块
import numpy as np                              # 导入 NumPy 库
import math, random                             # 导入数学和随机库
from DQN.environment import Env                 # 从 DQN 模块导入环境类
from DQN.DQNNet import Net, DQN                 # 从 DQNNet 模块导入网络结构和 DQN 类
import time                                     # 导入时间模块
import rospy                                    # 导入 ROS Python 库
from std_msgs.msg import Float32MultiArray      # 导入 ROS 中的 Float32MultiArray 消息类型
import os                                       # 导入操作系统模块
import sys                                      # 导入系统模块
from utils import plotLearning                  # 导入自定义绘图函数 plotLearning

# 添加项目路径，便于引入模块
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# 设置超参数
BATCH_SIZE = 512                                # 批次大小，训练时每次取出的样本数
N_ACTIONS = 5                                   # 动作空间大小（动作数量）
env = Env(N_ACTIONS)                            # 初始化环境，传入动作数量

# 主程序入口
if __name__=='__main__':

    dqn = DQN()                                              # 创建 DQN 实例（包括网络和记忆库等）
    rospy.init_node('limo_dqn')                              # 初始化 ROS 节点
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)        # 定义用于发布结果的 ROS 话题
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)# 定义用于发布动作的 ROS 话题
    result = Float32MultiArray()                             # 定义消息变量用于存储结果
    get_action = Float32MultiArray()                         # 定义消息变量用于存储动作
    start_time = time.time()                                 # 记录训练开始时间
    e = dqn.start_epoch                                      # 当前从哪一轮开始训练（可从中断处恢复）
    score_history = []                                       # 初始化一个列表用于记录每轮的得分
    save_figure_path = "/home/agilex/limo_ws/src/limoRL/train/DQN/png/"  # 设置保存图像路径
    past_action = 0                                          # 初始的前一个动作

    # 主循环，每一个 e 表示一个 episode（回合）
    for e in range(dqn.load_ep+1, 10000):

        s = env.reset(past_action)                           # 环境重置，返回初始状态
        episode_reward_sum = 0                               # 初始化本回合的总奖励
        done = False                                         # 表示是否结束
        episode_step = 6000                                  # 每个 episode 的最大步数

        # 内部循环，每一步进行动作选择、执行、学习等
        for t in range(episode_step):
            a = dqn.choose_action(s)                         # 使用当前策略选择一个动作
            s_, r, done = env.step(a, past_action)           # 执行动作，返回下一个状态、奖励和是否完成

            dqn.store_transition(s, a, r, s_)                # 存储这一步的样本进经验回放中
            episode_reward_sum += r                          # 累加本回合的奖励
            s = s_                                           # 更新当前状态
            pub_get_action.publish(get_action)              # 发布当前动作（用于可视化或其他节点）

            if dqn.memory_counter > BATCH_SIZE:              # 如果经验数量足够，则开始学习
                dqn.learn()                                  # 进行一次学习（更新网络）

            if e % 50 == 0:                                  # 每 50 轮保存一次模型
                dqn.save_model(str(e))                       # 保存模型

            if t >= 2500:                                    # 达到最大步数，强制终止该回合
                rospy.loginfo("time out!")
                done = True

            past_action = env.Getang(a)                      # 记录当前动作作为下一步使用

            # 打印当前步的信息，便于调试
            print("ep ", e)
            print("done ", done)
            print("reward ", r)
            print("action ", a)
            print("state ", s_)
            print("reward_sum ", episode_reward_sum)
            print("\n")

            if done:                                         # 如果回合结束，则跳出循环
                # 下面是用于可视化或记录结果，可以根据需要开启
                # result.data = [episode_reward_sum, float(dqn.loss), float(dqn.q_eval), float(dqn.q_target)]
                # pub_result.publish(result)
                # m, s = divmod(int(time.time() - start_time), 60)
                # h, m = divmod(m, 60)
                # rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                #               e, episode_reward_sum, dqn.memory_counter, dqn.epsilon, h, m, s)
                break

            # 每一步减小探索率 epsilon
            if dqn.epsilon > dqn.epsilon_min:
                dqn.epsilon = dqn.epsilon - 0.0001           # 逐步减小 epsilon 实现收敛探索

        score_history.append(episode_reward_sum)             # 记录每一回合的总得分

        if e % 2 == 0:                                       # 每隔两回合绘图一次
            filename_score = save_figure_path + "score_history_" + str(e) + ".png"  # 设置文件名
            plotLearning(score_history, filename_score, window=100)                 # 绘图并保存
            dqn.save_model(e=e)                              # 再次保存模型（可选）
            with open(save_figure_path + 'score_history.txt', 'w') as f:           # 将历史得分写入文件
                for score in score_history:
                    f.write(str(score) + '\n')
