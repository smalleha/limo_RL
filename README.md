# Limo 强化学习
环境[Ubuntu 20.04](https://zhida.zhihu.com/search?content_id=245782104&content_type=Article&match_order=1&q=Ubuntu+20.04&zhida_source=entity), [ROS1 Noetic](https://zhida.zhihu.com/search?content_id=245782104&content_type=Article&match_order=1&q=ROS1+Noetic&zhida_source=entity), python3.8 [pytorch 1.10](https://zhida.zhihu.com/search?content_id=245782104&content_type=Article&match_order=1&q=pytorch+1.10&zhida_source=entity)

训练的工控机：orin nano

## 使用方法

启动仿真

```
roslaunch limo_gazebo_sim limoEnv_diff.launch
```

启动强化学习训练

```
roslaunch limoRL limo_DDPG.launch
```

