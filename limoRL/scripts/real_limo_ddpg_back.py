#!/usr/bin/env python3
"""
DDPG 强化学习训练脚本，针对真实 Limo 平台，调用 Env Real Limo 库。
修复：动态获取状态维度，避免使用 AnyMsg。
"""
import os
import sys
import time
import copy
import math

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray

# 确保能找到 Env Real Limo 库
from DDPG.Real_Environment import Env
from DDPG.DDPGNet import Agent
from utils import plotLearning

NODE_NAME = 'limo_ddpg_real'

if __name__ == '__main__':
    # 离散动作维度（对应 Env.Getang 输出的角速度档数）
    ACTION_DIMENSION = 2
    ACTION_V_MAX = 0.5   # m/s
    ACTION_W_MAX = 1.5   # rad/s

    rospy.init_node(NODE_NAME)
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)

    # 创建真实车环境，先 reset 一次以获取状态维度
    env = Env(action_size=ACTION_DIMENSION)
    past_action = np.zeros(ACTION_DIMENSION)
    init_state = env.reset(past_action)
    state_dim = len(init_state)

    # 创建 DDPG Agent，使用动态计算的输入维度
    agent = Agent(
        alpha=0.000025,
        beta=0.00025,
        input_dims=[state_dim],
        tau=0.001,
        action_limit_v=ACTION_V_MAX,
        action_limit_w=ACTION_W_MAX,
        batch_size=64,
        layer1_size=256,
        layer2_size=128,
        n_actions=ACTION_DIMENSION
    )

    score_history = []

    for epoch in range(agent.start_epoch + 1, 50000):
        state = env.reset(past_action)
        episode_reward = 0.0
        done = False

        max_steps = 6000
        for t in range(max_steps):
            action = agent.choose_action(state)
            new_state, reward, done = env.step(action, past_action)
            episode_reward += reward

            agent.remember(state, action, reward, new_state, int(done))
            # agent.learn()

            # 发布可视化
            pub_get_action.publish(Float32MultiArray(data=action.tolist()))

            past_action = copy.deepcopy(action)
            state = copy.deepcopy(new_state)

            if t >= 1200:
                rospy.loginfo("Time out, end episode")
                done = True
            if episode_reward <= -2400:
                rospy.loginfo("Low reward, end episode")
                done = True

            if done:
                break

        score_history.append(episode_reward)
        pub_result.publish(Float32MultiArray(data=[episode_reward]))

        if epoch % 5 == 0:
            save_dir = os.path.expanduser("~/limo_ws/src/limoRL/train/DDPG/png")
            os.makedirs(save_dir, exist_ok=True)
            plotLearning(score_history, os.path.join(save_dir, f"score_{epoch}.png"), window=100)
            agent.save_models(index=epoch)

        rospy.loginfo(f"Epoch {epoch} Reward: {episode_reward:.2f}")
