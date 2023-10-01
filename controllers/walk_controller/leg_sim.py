from controller import Robot
from math import *
import numpy as np
import sys

from modern_robotics.ch04 import *
from modern_robotics.ch06 import *

class LegSim():

    def __init__(self, Slist, M, thetalist, motors):
        print('init')
        self.Slist = Slist
        self.M = M
        self.thetas = thetalist
        self.velcities = []
        self.torques = []
        self.motors = motors
        self.W = np.array([
            [0.8,   0,   0,   0,   0,   0],
            [  0, 0.8,   0,   0,   0,   0],
            [  0,   0, 0.8,   0,   0,   0],
            [  0,   0,   0, 0.6,   0,   0],
            [  0,   0,   0,   0, 0.6,   0],
            [  0,   0,   0,   0,   0, 0.6]
        ])
        for i in range(len(self.motors)):
            self.motors[i].set_position(self.thetas[i])

    def inverse_kinematics(self, T):
        # thetaL1, errL = IKinSpace(SL, ML, TL, thetaL, 0.01, 0.001)
        # thetaR1, errR = IKinSpace(SR, MR, TR, thetaR, 0.01, 0.001)
        # thetaL1, errL = IKinSpacePseudoInverse(SL, ML, TL, thetaL, 0.01, 0.001)
        # thetaR1, errR = IKinSpacePseudoInverse(SR, MR, TR, thetaR, 0.01, 0.001)
        # thetaL1, errL = IKinSpaceDampedLeastSquare1(SL, ML, TL, thetaL, 0.001, W, 0.01, 0.001)
        # thetaR1, errR = IKinSpaceDampedLeastSquare1(SR, MR, TR, thetaR, 0.001, W, 0.01, 0.001)
        # thetaL1, errL = IKinSpaceDampedLeastSquare2(SL, ML, TL, thetaL, 0.001, W, 0.1, 0.01, 0.001)
        # thetaR1, errR = IKinSpaceDampedLeastSquare2(SR, MR, TR, thetaR, 0.001, W, 0.1, 0.01, 0.001)
        # thetaL1, errL = IKinSpaceDampedPseudoInverse(SL, ML, TL, thetaL, 0.001, 0.8, 0.01, 0.001)
        # thetaR1, errR = IKinSpaceDampedPseudoInverse(SR, MR, TR, thetaR, 0.001, 0.8, 0.01, 0.001)
        # thetaL1, errL = IKinSpaceDamped(SL, ML, TL, thetaL, 0.5, 0.01, 0.001)
        # thetaR1, errR = IKinSpaceDamped(SR, MR, TR, thetaR, 0.5, 0.01, 0.001)
        thetas, result = IKinSpaceDampedLeastSquare1(self.Slist, self.M, T, self.thetas, 0.001, self.W, 0.01, 0.001)
        return thetas

    def position_control(self, T):
        self.thetas = self.inverse_kinematics(T)
        for i in range(len(self.motors)):
            self.motors[i].set_position(self.thetas[i])

    def velocity_control(self):
        for i in range(len(self.motors)):
            self.motors[i].set_velocity(self.velcities[i])

    def torque_control(self):
        for i in range(len(self.motors)):
            self.motors[i].set_torque(self.torques[i])
