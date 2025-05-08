cmake_minimum_required(VERSION 3.0.2)
project(limoRL)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)


catkin_package()


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(move src/move.cpp)
target_link_libraries(move
  ${catkin_LIBRARIES}
)
