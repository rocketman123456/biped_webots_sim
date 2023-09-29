"""walk_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Motor
from math import *
from random import random
import numpy as np
import sys

from walking.walking import *
from walking.preview_control import *
from motor_sim import *

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)
motor_L1 = robot.getDevice('Roll-L')
motor_L2 = robot.getDevice('Yaw-L')
motor_L3 = robot.getDevice('Pitch-L')
motor_L4 = robot.getDevice('Knee-L')
motor_L5 = robot.getDevice('Ankle-L')

motor_L1_dir = -1.0
motor_L2_dir = 1.0
motor_L3_dir = 1.0
motor_L4_dir = 1.0
motor_L5_dir = 1.0

sensor_L1 = robot.getDevice('Roll-L_sensor')
sensor_L2 = robot.getDevice('Yaw-L_sensor')
sensor_L3 = robot.getDevice('Pitch-L_sensor')
sensor_L4 = robot.getDevice('Knee-L_sensor')
sensor_L5 = robot.getDevice('Ankle-L_sensor')

motor_R1 = robot.getDevice('Roll-R')
motor_R2 = robot.getDevice('Yaw-R')
motor_R3 = robot.getDevice('Pitch-R')
motor_R4 = robot.getDevice('Knee-R')
motor_R5 = robot.getDevice('Ankle-R')

motor_R1_dir = -1.0
motor_R2_dir = 1.0
motor_R3_dir = -1.0
motor_R4_dir = -1.0
motor_R5_dir = -1.0

sensor_R1 = robot.getDevice('Roll-R_sensor')
sensor_R2 = robot.getDevice('Yaw-R_sensor')
sensor_R3 = robot.getDevice('Pitch-R_sensor')
sensor_R4 = robot.getDevice('Knee-R_sensor')
sensor_R5 = robot.getDevice('Ankle-R_sensor')

motor_sim_L1 = motor_sim(motor_L1, sensor_L1, motor_L1_dir, timestep)
motor_sim_L2 = motor_sim(motor_L2, sensor_L2, motor_L2_dir, timestep)
motor_sim_L3 = motor_sim(motor_L3, sensor_L3, motor_L3_dir, timestep)
motor_sim_L4 = motor_sim(motor_L4, sensor_L4, motor_L4_dir, timestep)
motor_sim_L5 = motor_sim(motor_L5, sensor_L5, motor_L5_dir, timestep)

motor_sim_R1 = motor_sim(motor_R1, sensor_R1, motor_R1_dir, timestep)
motor_sim_R2 = motor_sim(motor_R2, sensor_R2, motor_R2_dir, timestep)
motor_sim_R3 = motor_sim(motor_R3, sensor_R3, motor_R3_dir, timestep)
motor_sim_R4 = motor_sim(motor_R4, sensor_R4, motor_R4_dir, timestep)
motor_sim_R5 = motor_sim(motor_R5, sensor_R5, motor_R5_dir, timestep)

# motor = [motor_R1, motor_R2, motor_R3, motor_R4, motor_R5, motor_L1, motor_L2, motor_L3, motor_L4, motor_L5]

accelerometer = robot.getDevice('accelerometer')
accelerometer.enable(timestep)
gyro = robot.getDevice('gyro')
gyro.enable(timestep)

z = 0.3
left_foot = [0, 0, 0.3]
right_foot = [0, 0, 0.3]
joint_angles = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
joint_dirs = [1, -1, -1, -1, -1, 1, -1, -1, -1, -1]
name = "PAI"

pc = preview_control(0.01, 0.05, z)
walk = walking(left_foot, right_foot, joint_angles, pc, name)
foot_step = walk.setGoalPos([1.5, 0.0, 0])

# wait a moment
# for i in range(100):
    # robot.step(timestep)

j = 0
step = 0
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # val_L2 = sensor_L2.getValue()
    # val_R2 = sensor_R2.getValue()
    # acc = accelerometer.getValues()
    # w = gyro.getValues()
    # print(f"{val_L2}, {val_R2}")
    # print(f"{acc}, {w}")
    # j += 1
    # if j >= 10:
    #     joint_angles, lf, rf, xp, n = walk.getNextPos()
    #     # joint_angles_recorder = np.row_stack((joint_angles_recorder, np.array(joint_angles)))
    #     # print(f'\n-> {step}_joint_angles is:{joint_angles}/{lf}/{rf}/{n}\n')
    #     j = 0
    #     if n == 0:
    #         if (len(foot_step) <= 5):
    #             x_goal, y_goal, th = random()-0.5, random()-0.5, random()-0.5
    #             # break
    #             foot_step = walk.setGoalPos([x_goal, y_goal, th])
    #         else:
    #             # print("\n\n ==n is null==\n")
    #             foot_step = walk.setGoalPos()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    # for i in range(10):
    #     motor[i].setPosition(joint_dirs[i] * joint_angles[i])
    # motor_R5.setPosition(0.5)
    motor_sim_L3.setPosition(-0.4)
    motor_sim_L4.setPosition(0.8)
    motor_sim_L5.setPosition(-0.4)

    motor_sim_R3.setPosition(-0.4)
    motor_sim_R4.setPosition(0.8)
    motor_sim_R5.setPosition(-0.4)

    # print(f"{motor_sim_L5.getTorque()}, {motor_sim_R5.getTorque()}")

    step += 1
    pass

# Enter here exit cleanup code.
