#!/usr/bin/env python3

import numpy as np                                                               
import math
import time
import rospy
from std_msgs.msg import Float32MultiArray
import os
import sys
import copy
from TD3.TD3Net import TD3
from TD3.Environment import Env
from utils import plotLearning
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


if __name__=='__main__':
    
    PI = math.pi
    ACTION_DIMENSION = 2
    ACTION_V_MAX = 0.8 # m/s
    ACTION_W_MAX = 2.0 # rad/s
    CKPT_DIR = "/home/agilex/limo_ws/src/limoRL/train/TD3/model"
    
    agent = TD3(alpha=0.0003, beta=0.0003, state_dim=25,action_dim=2, actor_fc1_dim=400, actor_fc2_dim=300,
                action_limit_v=ACTION_V_MAX,action_limit_w=ACTION_W_MAX,critic_fc1_dim=400, critic_fc2_dim=300,
                ckpt_dir=CKPT_DIR, gamma=0.99,tau=0.005, action_noise=0.1, policy_noise=0.2, policy_noise_clip=0.5,
                delay_time=2, max_size=10000000, batch_size=512)

    rospy.init_node('limo_td3')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    start_time =time.time()
    
    env = Env(action_dim=ACTION_DIMENSION)
    past_action = np.zeros(ACTION_DIMENSION)
    save_figure_path = "/home/agilex/limo_ws/src/limoRL/train/TD3/png/"
    score_history = []

    
    # Load model 
    #agent.load_models(episode=agent.start_epoch)
    for e in range(agent.start_epoch+1,50000):
        
        state = env.reset()      #获得当前的状态
        episode_reward_sum = 0   #当前回合的奖励
        done = False             #是否回合结束                       
        episode_step=6000        
        for t in range(episode_step):
            action = agent.choose_action(state,train=True)          #选择新的动作
            new_state,reward,done = env.step(action,past_action)    #获得新的状态 奖励 及其判断是否结束当前回合
            agent.remember(state, action, reward, new_state, done)  #存放记忆库
            agent.learn()                                           #进行强化学习训练
            episode_reward_sum +=reward
            past_action = copy.deepcopy(action)
            state = copy.deepcopy(new_state)
            
            print("ep ",e)
            print("done ",done)
            print("reward ",reward)
            print("action ",action)
            print("state ",state)
            print("reward_sum ",episode_reward_sum)
            print("v = ",action[0])
            print("w = ",action[1])
            print("\n")        

            if t >=1200:
                rospy.loginfo("time out!")
                done = True
            
            if episode_reward_sum <= -2400:
                rospy.loginfo("reward fail")
                done = True

            if done:
                print("after done ")
                break

        
        # 保存模型 及其 奖励曲线
        score_history.append(episode_reward_sum)
        if e % 5 == 0:
            filename_score = save_figure_path + "score_history_" + str(e) + ".png"
            plotLearning(score_history, filename_score, window=100)
            agent.save_models(episode=e)
            with open(save_figure_path +'score_history.txt', 'w') as f:
                for score in score_history:
                    f.write(str(score) + '\n')
