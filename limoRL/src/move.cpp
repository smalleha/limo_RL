#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
 
/*
    需求：发布速度消息  
            话题：/turtle1/cmd_vel
            消息：geometry_msgs/Twist
        1 包含头文件
        2 初始化ros节点
        3 创建节点句柄
        4 创建发布对象
        5 发布逻辑实现
        6 回旋函数【可选】 spinOnce()
*/
 
 
int main(int argc, char  *argv[])
{
        setlocale(LC_ALL,"");
 
       // 2 初始化ros节点
        ros::init(argc,argv,"my_control");
 
       // 3 创建节点句柄
       ros::NodeHandle nh;
 
       // 4 创建发布对象
       ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel",10);
 
       // 5 发布逻辑实现
        ros::Rate rate(10);             //设置发布频率
        //组织被发布的消息
        geometry_msgs::Twist twist;
        //线速度
        twist.linear.x = 0.;     //浮点类型
        twist.linear.y = 0.0 ;
        twist.linear.z = 0.0;
 
        //角速度
        twist.angular.x = 0.0;
        twist.angular.y =0.0;
        twist.angular.z = 0.;

        printf("11");
        //循环发布
        while(ros::ok())
        {
            printf("22");
            pub.publish(twist);
            //休眠
            rate.sleep();
            // 6 回旋函数【可选】 spinOnce()
            ros::spinOnce();
        }
    return 0;
}