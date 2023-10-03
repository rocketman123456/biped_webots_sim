from controller import Robot, Motor
from math import *
import numpy as np
import sys

from modern_robotics.ch04 import *
from modern_robotics.ch06 import *
from motor_sim import *
from leg_sim import *
from state_etimation import *

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:

def motor_define():
    motor_L1 = robot.getDevice('Roll-L')
    motor_L2 = robot.getDevice('Yaw-L')
    motor_L3 = robot.getDevice('Pitch-L')
    motor_L4 = robot.getDevice('Knee-L')
    motor_L5 = robot.getDevice('Ankle-L')

    sensor_L1 = robot.getDevice('Roll-L_sensor')
    sensor_L2 = robot.getDevice('Yaw-L_sensor')
    sensor_L3 = robot.getDevice('Pitch-L_sensor')
    sensor_L4 = robot.getDevice('Knee-L_sensor')
    sensor_L5 = robot.getDevice('Ankle-L_sensor')

    motor_L1_dir = -1.0
    motor_L2_dir = 1.0
    motor_L3_dir = 1.0
    motor_L4_dir = 1.0
    motor_L5_dir = 1.0

    motor_R1 = robot.getDevice('Roll-R')
    motor_R2 = robot.getDevice('Yaw-R')
    motor_R3 = robot.getDevice('Pitch-R')
    motor_R4 = robot.getDevice('Knee-R')
    motor_R5 = robot.getDevice('Ankle-R')

    sensor_R1 = robot.getDevice('Roll-R_sensor')
    sensor_R2 = robot.getDevice('Yaw-R_sensor')
    sensor_R3 = robot.getDevice('Pitch-R_sensor')
    sensor_R4 = robot.getDevice('Knee-R_sensor')
    sensor_R5 = robot.getDevice('Ankle-R_sensor')

    motor_R1_dir = -1.0
    motor_R2_dir = 1.0
    motor_R3_dir = -1.0
    motor_R4_dir = -1.0
    motor_R5_dir = -1.0

    motor_sim_L1 = MotorSim(motor_L1, sensor_L1, motor_L1_dir, timestep)
    motor_sim_L2 = MotorSim(motor_L2, sensor_L2, motor_L2_dir, timestep)
    motor_sim_L3 = MotorSim(motor_L3, sensor_L3, motor_L3_dir, timestep)
    motor_sim_L4 = MotorSim(motor_L4, sensor_L4, motor_L4_dir, timestep)
    motor_sim_L5 = MotorSim(motor_L5, sensor_L5, motor_L5_dir, timestep)

    motor_sim_R1 = MotorSim(motor_R1, sensor_R1, motor_R1_dir, timestep)
    motor_sim_R2 = MotorSim(motor_R2, sensor_R2, motor_R2_dir, timestep)
    motor_sim_R3 = MotorSim(motor_R3, sensor_R3, motor_R3_dir, timestep)
    motor_sim_R4 = MotorSim(motor_R4, sensor_R4, motor_R4_dir, timestep)
    motor_sim_R5 = MotorSim(motor_R5, sensor_R5, motor_R5_dir, timestep)

    motorsL = [motor_sim_L1, motor_sim_L2, motor_sim_L3, motor_sim_L4, motor_sim_L5]
    motorsR = [motor_sim_R1, motor_sim_R2, motor_sim_R3, motor_sim_R4, motor_sim_R5]

    return motorsL, motorsR

motorsL, motorsR = motor_define()

##############################################################################
##############################################################################
##############################################################################

def leg_define():
    L0 = 0.25 / 2.0
    L1 = 0.25
    L2 = 0.25
    L3 = 0.02 # 0.045

    SL1 = np.array([1, 0, 0, 0, 0, 0])
    SL2 = np.array([0, 0, 1, 0, 0, 0])
    SL3 = np.array([0, 1, 0, 0, 0, 0])
    SL4 = np.array([0, 1, 0, L1, 0, 0])
    SL5 = np.array([0, 1, 0, L1+L2, 0, 0])

    SL = np.array([SL1, SL2, SL3, SL4, SL5]).T

    ML = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -(L1 + L2 + L3)],
        [0, 0, 0, 1]
    ])

    thetaL = np.array([0, 0, -0.3, 0.6, -0.3])

    SR1 = np.array([1, 0, 0, 0, 0, 0])
    SR2 = np.array([0, 0, 1, 0, 0, 0])
    SR3 = np.array([0, 1, 0, 0, 0, 0])
    SR4 = np.array([0, 1, 0, L1, 0, 0])
    SR5 = np.array([0, 1, 0, L1+L2, 0, 0])

    SR = np.array([SR1, SR2, SR3, SR4, SR5]).T

    MR = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -(L1 + L2 + L3)],
        [0, 0, 0, 1]
    ])

    thetaR = np.array([0, 0, -0.3, 0.6, -0.3])

    W = np.array([
        [0.1,   0,   0,   0,   0,   0],
        [  0, 0.1,   0,   0,   0,   0],
        [  0,   0, 0.1,   0,   0,   0],
        [  0,   0,   0, 0.4,   0,   0],
        [  0,   0,   0,   0, 0.4,   0],
        [  0,   0,   0,   0,   0, 0.4]
    ])

    legL = LegSim(SL, ML, thetaL, motorsL)
    legR = LegSim(SR, MR, thetaR, motorsR)

    return legL, legR

legL, legR = leg_define();

##############################################################################
##############################################################################
##############################################################################

accelerometer = robot.getDevice('accelerometer')
accelerometer.enable(timestep)

gyro = robot.getDevice('gyro')
gyro.enable(timestep)

z = 0.4

TL = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, -z],
    [0, 0, 0, 1]
])

TR = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, -z],
    [0, 0, 0, 1]
])

##############################################################################
##############################################################################
##############################################################################

legL.position_control(TL)
legR.position_control(TR)

# wait a moment
for i in range(100):
    robot.step(timestep)

count = 0
angle = 0
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    acc = accelerometer.getValues()
    w = gyro.getValues()

    # Process sensor data here.
    angle += 0.1
    count += 1

    # Enter here functions to send actuator commands, like:
    # TL = np.array([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, -z + 0.1 * sin(angle)],
    #     [0, 0, 0, 1]
    # ])

    # TR = np.array([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, -z + 0.1 * sin(angle)],
    #     [0, 0, 0, 1]
    # ])

    if count < 100:
        legL.torque_control(np.array([0, 0, 0, 0, 0, -15]).T)
        legR.torque_control(np.array([0, 0, 0, 0, 0, -15]).T)
    else:
        legL.position_control(TL)
        legR.position_control(TR)

    pass

# Enter here exit cleanup code.
