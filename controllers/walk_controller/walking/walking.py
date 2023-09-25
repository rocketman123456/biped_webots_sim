#!/usr/bin/env python3
#
# generating walking pattern for the GankenKun

import pybullet as p
import numpy as np
import csv
import time
from mos_leg_ik import *
from foot_step_planner import *

class walking():
    def __init__(self, left_foot0, right_foot0, joint_angles, pc, name="MOS"):
        self.kine = THMOSLegIK(name)
        self.foot_width = 0.06
        if name == "MOS":
            self.foot_width = 0.06
        elif name == "PAI":
            self.foot_width = 0.1
        self.left_foot0, self.right_foot0 = left_foot0, right_foot0
        self.joint_angles = joint_angles
        self.pc = pc
        self.fsp = foot_step_planner(0.05, 0.03, 0.2, 0.3, self.foot_width)  # period 0.34
        self.X = np.matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        self.pattern = []
        self.left_up = self.right_up = 0.0
        self.left_off,  self.left_off_g,  self.left_off_d = np.matrix([[0.0, 0.0, 0.0]]),  np.matrix([[0.0, 0.0, 0.0]]),  np.matrix([[0.0, 0.0, 0.0]])
        self.right_off, self.right_off_g, self.right_off_d = np.matrix([[0.0, 0.0, 0.0]]),  np.matrix([[0.0, 0.0, 0.0]]),  np.matrix([[0.0, 0.0, 0.0]])
        self.th = 0
        self.status = 'start'
        self.next_leg = 'right'
        self.foot_step = []
        self.name = name
        return

    def setGoalPos(self, pos=None):
        if pos == None:
            if len(self.foot_step) <= 4:
                self.status = 'start'
            if len(self.foot_step) > 3:
                del self.foot_step[0]
        else:
            if len(self.foot_step) > 2:
                if not self.status == 'start':
                    offset_y = -self.foot_width if self.next_leg == 'left' else self.foot_width
                else:
                    offset_y = 0.0
                current_x, current_y, current_th = self.foot_step[1][1], self.foot_step[1][2] + offset_y, self.foot_step[1][3]
            else:
                current_x, current_y, current_th = 0, 0, 0
            # print("\n# fsp.calculate:",pos[0], pos[1], pos[2], current_x, current_y, current_th, self.next_leg, self.status)
            self.foot_step = self.fsp.calculate(pos[0], pos[1], pos[2], current_x, current_y, current_th, self.next_leg, self.status)
            self.status = 'walking'
        # print("\n#1. foot_step is: ", str(self.foot_step))

        t = self.foot_step[0][0]
        self.pattern, x, y = self.pc.set_param(t, self.X[:, 0], self.X[:, 1], self.foot_step)
        # print("\n# COG_X is: ", len(self.pattern), self.pattern)
        # print("\n# x is: ", x)
        # print("\n# y is: ", y)
        # print("\n#1. X is:", self.X)

        self.X = np.matrix([[x[0, 0], y[0, 0]], [x[1, 0], y[1, 0]], [x[2, 0], y[2, 0]]])
        # print("\n#2. X is:", self.X)
        if self.foot_step[0][4] == 'left':
            if self.foot_step[1][4] == 'both':
                self.right_off_g = np.matrix([[self.foot_step[1][1], self.foot_step[1][2], self.foot_step[1][3]]])
            else:
                self.right_off_g = np.matrix(
                    [[self.foot_step[1][1], self.foot_step[1][2]+self.foot_width, self.foot_step[1][3]]])
            self.right_off_d = (self.right_off_g - self.right_off)/17.0
            self.next_leg = 'right'
        if self.foot_step[0][4] == 'right':
            if self.foot_step[1][4] == 'both':
                self.left_off_g = np.matrix([[self.foot_step[1][1], self.foot_step[1][2], self.foot_step[1][3]]])
            else:
                self.left_off_g = np.matrix([[self.foot_step[1][1], self.foot_step[1][2]-self.foot_width, self.foot_step[1][3]]])
            self.left_off_d = (self.left_off_g - self.left_off)/17.0
            self.next_leg = 'left'
        self.th = self.foot_step[0][3]
        # print("\n#2. foot_step is: ", str(self.foot_step))
        return self.foot_step

    def getNextPos(self):
        t1 = time.time()
        X = self.pattern.pop(0)
        period = round((self.foot_step[1][0]-self.foot_step[0][0])/0.01)
        self.th += (self.foot_step[1][3]-self.foot_step[0][3])/period
        x_dir = 0
        BOTH_FOOT = round(0.17/0.01)
        start_up = round(BOTH_FOOT/2)
        # end_up   = round(period/2)
        end_up = round(period/2)
        period_up = end_up - start_up
        period_down = period-end_up
        foot_hight = 0.05

        if self.foot_step[0][4] == 'right':
            # up or down foot
            if start_up < (period-len(self.pattern)) <= end_up:
                self.left_up += foot_hight/period_up
            elif self.left_up > 0:
                self.left_up = max(self.left_up - foot_hight/period_down, 0.0)
            # move foot in the axes of x,y,the
            if (period-len(self.pattern)) > start_up:
                self.left_off += self.left_off_d
                if (period-len(self.pattern)) > (start_up + period_up * 2):
                    self.left_off = self.left_off_g.copy()

        if self.foot_step[0][4] == 'left':
            # up or down foot
            if start_up < (period-len(self.pattern)) <= end_up:
                self.right_up += foot_hight/period_up
            elif self.right_up > 0:
                self.right_up = max(self.right_up - foot_hight/period_down, 0.0)
            # move foot in the axes of x,y,the
            if (period-len(self.pattern)) > start_up:
                self.right_off += self.right_off_d
                if (period-len(self.pattern)) > (start_up + period_up * 2):
                    self.right_off = self.right_off_g.copy()

        lo = self.left_off - np.block([[X[0, 0:2], 0]])
        ro = self.right_off - np.block([[X[0, 0:2], 0]])

        # print(f"\n# lo:{lo}\n# ro:{ro}\n")

        # left_foot  = [self. left_foot0[0]+lo[0,0]-0.077, self. left_foot0[1]+lo[0,1]-0.05, self. left_foot0[2]-self.left_up -0.43-0.21, 0.0, 0.0, self.th-lo[0,2]]
        # left_foot  = [self. left_foot0[0]+lo[0,0]-0.077, self.left_foot0[1]+lo[0,1]-0.05, self.left_up-0.3, 0.0, 0.0, self.th-lo[0,2]]
        # right_foot = [self.right_foot0[0]+ro[0,0]-0.077, self.right_foot0[1]+ro[0,1]+0.05, self.right_up-0.3, 0.0, 0.0, self.th-ro[0,2]]
        left_foot = [self. left_foot0[0]+lo[0, 0]-0.047, self.left_foot0[1] + lo[0, 1]-0.05, self.left_up-0.28, 0.0, 0.0, self.th-lo[0, 2]]
        right_foot = [self.right_foot0[0]+ro[0, 0]-0.047, self.right_foot0[1] + ro[0, 1]+0.05, self.right_up-0.28, 0.0, 0.0, self.th-ro[0, 2]]

        # print("\n# ", self.foot_step[0][4])
        # print(f"\n# left_foot:{left_foot}\n# right_foot:{right_foot}\n")

        l_joint_angles = self.kine.LegIKMove('left', left_foot)
        r_joint_angles = self.kine.LegIKMove('right', right_foot)
        if self.name == "MOS":
            self.joint_angles = r_joint_angles+l_joint_angles  # R first
        elif self.name == "PAI":
            self.joint_angles = l_joint_angles+r_joint_angles  # R first

        # print(f"\n# left_foot:{l_joint_angles}\n# right_foot:{r_joint_angles}\n")

        xp = [X[0, 2], X[0, 3]]

        return self.joint_angles, left_foot, right_foot, xp, len(self.pattern)
