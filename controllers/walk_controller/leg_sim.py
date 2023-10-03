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
        self.dthetas = []
        self.ddthetas = []
        self.torques = []
        self.Glist = []
        self.motors = motors
        self.W = np.array([
            [0.8,   0,   0,   0,   0,   0],
            [0, 0.8,   0,   0,   0,   0],
            [0,   0, 0.8,   0,   0,   0],
            [0,   0,   0, 0.6,   0,   0],
            [0,   0,   0,   0, 0.6,   0],
            [0,   0,   0,   0,   0, 0.6]
        ])
        for i in range(len(self.motors)):
            self.motors[i].set_position(self.thetas[i])

    def inverse_kinematics(self, T):
        # thetas, result = IKinSpace(self.Slist, self.M, T, self.thetas, 0.01, 0.001)
        # thetas, result = IKinSpacePseudoInverse(self.Slist, self.M, T, self.thetas, 0.01, 0.001)
        # thetas, result = IKinSpaceDamped(self.Slist, self.M, T, self.thetas, 0.5, 0.01, 0.001)
        # thetas, result = IKinSpaceDampedPseudoInverse(self.Slist, self.M, T, self.thetas, 0.001, 0.8, 0.01, 0.001)
        thetas, result = IKinSpaceDampedLeastSquare1(self.Slist, self.M, T, self.thetas, 0.001, self.W, 0.01, 0.001)
        # thetas, result = IKinSpaceDampedLeastSquare2(self.Slist, self.M, T, self.thetas, 0.001, self.W, 0.1, 0.01, 0.001)
        return thetas

    def inverse_velocity(self, Vs):
        # Tsb = FKinSpace(M, Slist, thetalist)
        # Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
        dthetas = np.linalg.pinv(JacobianSpace(self.Slist, self.thetas)) @ Vs
        return dthetas

    def inverse_force(self, Fs):
        # Tsb = FKinSpace(M, Slist, thetalist)
        # Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
        torques = JacobianSpace(self.Slist, self.thetas).T @ Fs
        return torques

    def position_control(self, T):
        self.thetas = self.inverse_kinematics(T)
        for i in range(len(self.motors)):
            self.motors[i].set_position(self.thetas[i])

    def velocity_control(self, Vs):
        self.dthetas = self.inverse_velocity(Vs)
        # print(self.dthetas)
        for i in range(len(self.motors)):
            self.motors[i].set_velocity(self.dthetas[i])

    def force_control(self, Fs):
        self.torques = self.inverse_force(Fs)
        # print(self.torques)
        for i in range(len(self.motors)):
            self.motors[i].set_torque(self.torques[i])
