"""walk_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from math import *

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

sensor_R1 = robot.getDevice('Roll-R_sensor')
sensor_R2 = robot.getDevice('Yaw-R_sensor')
sensor_R3 = robot.getDevice('Pitch-R_sensor')
sensor_R4 = robot.getDevice('Knee-R_sensor')
sensor_R5 = robot.getDevice('Ankle-R_sensor')

sensor_L1.enable(timestep)
sensor_L2.enable(timestep)
sensor_L3.enable(timestep)
sensor_L4.enable(timestep)
sensor_L5.enable(timestep)

sensor_R1.enable(timestep)
sensor_R2.enable(timestep)
sensor_R3.enable(timestep)
sensor_R4.enable(timestep)
sensor_R5.enable(timestep)

angle = 0
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    val_L2 = sensor_L2.getValue()
    val_R2 = sensor_R2.getValue();
    print(f"{val_L2}, {val_R2}")

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    angle += 0.01
    motor_L2.setPosition(sin(angle))
    motor_R2.setPosition(-sin(angle))
    pass

# Enter here exit cleanup code.
