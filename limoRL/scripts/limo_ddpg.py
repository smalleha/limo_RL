#!/usr/bin/env python

import numpy as np                              # 导入numpy                                   
import math
from DDPG.Environment import Env
# from DDPG.Real_Environment import Env
from DDPG.DDPGNet import Agent
import time
import rospy
from std_msgs.msg import Float32MultiArray
import os
import sys
import copy
from utils import plotLearning
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

if __name__=='__main__':
    
    ACTION_DIMENSION = 2
    ACTION_V_MAX = 0.5 # m/s
    ACTION_W_MAX = 1.5 # rad/s
    PI = math.pi
    
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[25], tau=0.001, action_limit_v =ACTION_V_MAX ,
                  action_limit_w = ACTION_W_MAX,batch_size=64, layer1_size=256, layer2_size=128, n_actions = ACTION_DIMENSION)
    rospy.init_node('limo_ddpg')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    start_time =time.time()
    
    
    env = Env(action_dim=ACTION_DIMENSION)
    past_action = np.zeros(ACTION_DIMENSION)
    save_figure_path = "/home/agilex/limo_ws/src/limoRL/train/DDPG/png/"
    score_history = []
    actor_history = []
    critic_history = []
    
    # load model
    # agent.load_models(index=440)
    for e in range(agent.start_epoch+1,50000):
        
        state = env.reset()
        episode_reward_sum = 0
        done = False
        episode_step=6000
        
        for t in range(episode_step):
            action = agent.choose_action(state)   
            new_state,reward,done = env.step(action,past_action)
            episode_reward_sum +=reward
            agent.remember(state, action, reward, new_state, int(done))
            # agent.learn()
            past_action = copy.deepcopy(action)
            state = copy.deepcopy(new_state)
            
            print("ep ",e)
            print("done ",done)
            print("reward ",reward)
            print("action ",action)
            print("state ",state)
            print("reward_sum ",episode_reward_sum)
            print("\n")
            
            if t >=1200:
                rospy.loginfo("time out!")
                done =True

            if episode_reward_sum <= -2400:
                rospy.loginfo("reward fail")
                done = True
        
            if done:
                print("after done ")
                break


        score_history.append(episode_reward_sum)
        if e % 5 == 0:
            filename_score = save_figure_path + "score_history_" + str(e) + ".png" 
            plotLearning(score_history, filename_score, window=100)
            agent.save_models(index=e)


            
   
            
    
    
  