from controller import Robot
from math import *
import numpy as np
import sys


class Leg():

    def __init__(self):
        print('init')
        self.Slist = []
        self.M = []
        self.thetalist = []
        self.motors = []

    def inverse_kinematics(self):
        print('inverse_kinematics')

    def position_control(self):
        print('position')

    def velocity_control(self):
        print('velocity')

    def torque_control(self):
        print('torque')
