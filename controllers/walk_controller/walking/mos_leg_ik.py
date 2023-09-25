#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright (c). All Rights Reserved.
# -----------------------------------------------------
# File Name:        THMOSIKPacket.py
# Creator:          JinYin Zhou
# Version:          0.1
# Created:          2023/2/20
# Description:      leg ik packet
# Function List:        class LegIK:
# History:
#   <author>      <version>       <time>          <description>
#   Jinyin Zhou     0.1           2023/2/20       create
# -----------------------------------------------------

import math
import numpy as np


class LegIK:
    """inverse kinematics for one leg"""

    def __init__(self, LeftorRight, Legs_len, motor_offset, motor_way, name="MOS"):
        """
        initialize class
        Args:
            LeftorRight :id-left leg or right leg string-‘left’ 'right' 
            Legs_len : a list of leg len [3]
            motor_offset : a list of offset angle [6]
            motor_way : a list of spin direction of motor [6,+1,-1]
        """
        self.LorR = LeftorRight
        self.legs_len = Legs_len
        self.offset = motor_offset
        self.way = motor_way
        self.name = name
        self.theta = [0] * 6

    def RtoRPY(self, R):
        '''rotate matrix to RPY angles'''
        R = np.array(R)
        err = float(0.001)
        oy = math.atan2(-R[2, 0], math.sqrt((R[0, 0])**2 + (R[1, 0])**2))

        if oy >= math.pi/2-err and oy <= math.pi/2+err:
            oy = math.pi/2
            oz = 0.0
            ox = math.atan2(R[0, 1], R[0, 2])
        elif oy >= -(math.pi/2)-err and oy <= -(math.pi/2)+err:
            oy = -math.pi/2
            oz = 0.0
            ox = math.atan2(-R[0, 1], -R[0, 2])
        else:
            oz = math.atan2((R[1, 0])/(math.cos(oy)), (R[0, 0])/(math.cos(oy)))
            ox = math.atan2((R[2, 1])/(math.cos(oy)), (R[2, 2])/(math.cos(oy)))

        return [ox, oy, oz]

    def RPYtoR(self, rpy):
        '''RPY angles to rotate matrix'''
        a = rpy[0]
        b = rpy[1]
        c = rpy[2]

        sinA = np.sin(a)
        cosA = np.cos(a)
        sinB = np.sin(b)
        cosB = np.cos(b)
        sinC = np.sin(c)
        cosC = np.cos(c)

        R = [[cosB*cosC,  cosC*sinA*sinB - cosA*sinC,  sinA*sinC + cosA*cosC*sinB],
             [cosB*sinC,  cosA*cosC + sinA*sinB*sinC, cosA*sinB*sinC - cosC*sinA],
             [-sinB, cosB*sinA,  cosA*cosB]]
        return R

    def getTfromRd(self, R, d):
        T = np.identity(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = d
        return T

    def se3toSE3(self, se3, theta):
        '''torch to transform [numpy.array]'''
        SO3 = np.mat([[0,      -se3[2], se3[1]],
                      [se3[2], 0,      -se3[0]],
                      [-se3[1], se3[0], 0]])
        R = np.mat(np.identity(3)) + np.sin(theta) * SO3 + (1 - np.cos(theta)) * SO3 * SO3
        d = theta * np.mat(np.identity(3)) + (1 - np.cos(theta)) * SO3 + np.sin(theta) * SO3 + (theta - np.sin(theta)) * SO3 * SO3
        d = np.dot(d, np.array(se3[3:6]))
        return self.getTfromRd(R, d)

    def getRfromT(self, T):
        '''get rotation matrix from transform'''
        T = np.array(T)
        R = T[0:3, 0:3]
        return R

    def getdfromT(self, T):
        '''get vector from transform'''
        T = np.array(T)
        d = T[0:3, 3]
        return d

    def StdLegIK(self, end_point, end_rpy):
        '''inverse kinematics with standard axis'''

        # get foot target transformation
        Rt = np.array(self.RPYtoR(end_rpy))
        dt = np.array(end_point)
        T = self.getTfromRd(Rt, dt)

        # get inverse to change coordinates(from lab to foot)
        iT = np.linalg.inv(T)

        # caculate self.theta 3 4 5
        d = self.getdfromT(iT)
        la = np.linalg.norm(d - np.array([0, 0, self.legs_len[2]]))

        # 3 4
        if (abs(self.legs_len[0] - self.legs_len[1]) > la):
            self.theta[3] = - np.pi
            theta_a = np.pi
        elif (self.legs_len[0] + self.legs_len[1] > la):
            self.theta[3] = np.arccos((self.legs_len[0] ** 2 + self.legs_len[1] ** 2 - la ** 2) / (2 * self.legs_len[0] * self.legs_len[1])) - np.pi
            theta_a = np.arccos((self.legs_len[1] ** 2 + la ** 2 - self.legs_len[0] ** 2) / (2 * self.legs_len[1] * la))
        else:
            self.theta[3] = 0
            theta_a = 0
        self.theta[4] = theta_a + np.arcsin(d[0] / la)

        # 5
        self.theta[5] = np.arctan(- d[1] / (d[2] - self.legs_len[2]))

        # caculate self.theta 0 1 2
        T5 = np.mat(self.se3toSE3([1, 0, 0, 0, 0, self.legs_len[2]], self.theta[5]))
        T4 = np.mat(self.se3toSE3([0, 1, 0, 0, 0, self.legs_len[2]], self.theta[4]))
        T3 = np.mat(self.se3toSE3([0, 1, 0, 0, 0, self.legs_len[2] + self.legs_len[1]], self.theta[3]))
        Tfoot = T5 * T4 * T3
        Tlap = Tfoot.I * np.mat(iT)
        self.theta[0:3] = self.RtoRPY(self.getRfromT(Tlap))

        # set theta
        if (self.name == "MOS"):
            for index in range(6):
                self.theta[index] = self.theta[index] * self.way[index] + self.offset[index]

        elif (self.name == "PAI"):
            theta = [0] * 5
            for index in range(5):
                self.theta[index] = self.theta[index] * self.way[index] + self.offset[index]
                if index == 1:
                    theta[2] = self.theta[1]
                elif index == 2:
                    theta[1] = self.theta[2]
                elif index < 5:
                    theta[index] = self.theta[index]
            return theta

        return self.theta


class THMOSLegIK:
    """inverse kinematics for two leg"""

    def __init__(self, name="MOS"):
        """
        initialize class
        """
        self.name = name
        if name == "MOS":
            self.leg_left = LegIK('Left', [0.156, 0.12, 0.045], [0, 0, 0, 0, 0, 0], [1, -1, -1, -1, -1, -1], name)
            self.leg_right = LegIK('Right', [0.156, 0.12, 0.045], [0, 0, 0, 0, 0, 0], [1, 1, -1, 1, 1, -1], name)
        if name == "PAI":
            self.leg_left = LegIK('Left', [0.16, 0.16, 0.03], [0, 0, 0, 0, 0, 0], [1, -1, -1, -1, -1, -1], name)
            self.leg_right = LegIK('Right', [0.16, 0.16, 0.03], [0, 0, 0, 0, 0, 0], [1, 1, -1, 1, 1, -1], name)

    def LegIKMove(self, LeftorRight, end_pos, body_rpy=[0, 0, 0]):
        """
        move left and right leg
        """
        xyz = end_pos[0:3]
        real_rpy = end_pos[3:6]
        Rworld_to_body = np.mat(self.leg_left.RPYtoR(body_rpy))
        Rworld_to_feet = np.mat(self.leg_left.RPYtoR(real_rpy))
        # body to feet
        rpy = self.leg_left.RtoRPY(Rworld_to_body.T * Rworld_to_feet)

        if (LeftorRight == 'Left' or LeftorRight == 'left'):
            return self.leg_left.StdLegIK(xyz, rpy)
        else:
            return self.leg_right.StdLegIK(xyz, rpy)
