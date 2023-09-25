#!/usr/bin/env python3
#
# solving kinematics for the SUSTAINA-OP

import math
import pybullet as p


class kinematics():
    L1 = 0.100
    L12 = 0.057
    L2 = 0.100
    L3 = 0.053
    OFFSET_W = 0.044
    OFFSET_X = 0.0

    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.index_dof = {p.getBodyInfo(self.robot_id)[0].decode('UTF-8'): -1, }
        for id in range(p.getNumJoints(self.robot_id)):
            self.index_dof[p.getJointInfo(self.robot_id, id)[12].decode('UTF-8')] = p.getJointInfo(self.robot_id, id)[3] - 7

    def solve_ik(self, left_foot, right_foot, current_angles):
        joint_angles = current_angles.copy()
        l_x, l_y, l_z, l_roll, l_pitch, l_yaw = left_foot
        l_x -= self.OFFSET_X
        l_y -= self.OFFSET_W
        l_z = self.L1 + self.L12 + self.L2 + self.L3 - l_z
        l_x2 = l_x * math.cos(l_yaw) + l_y * math.sin(l_yaw)
        l_y2 = -l_x * math.sin(l_yaw) + l_y * math.cos(l_yaw)
        l_z2 = l_z - self.L3
        waist_roll = math.atan2(l_y2, l_z2)
        l2 = l_y2**2 + l_z2**2
        l_z3 = math.sqrt(max(l2 - l_x2**2, 0.0)) - self.L12
        pitch = math.atan2(l_x2, l_z3)
        l = math.sqrt(l_x2**2 + l_z3**2)
        knee_disp = math.acos(min(max(l/(2.0*self.L1), -1.0), 1.0))
        waist_pitch = - pitch - knee_disp
        knee_pitch = - pitch + knee_disp
        joint_angles[self.index_dof['left_waist_yaw_link']] = l_yaw
        joint_angles[self.index_dof['left_waist_roll_link']] = waist_roll
        joint_angles[self.index_dof['left_waist_pitch_link']] = waist_pitch
        joint_angles[self.index_dof['left_knee_pitch_link']] = -waist_pitch
        joint_angles[self.index_dof['left_waist_pitch_mimic_link']] = waist_pitch
        joint_angles[self.index_dof['left_shin_pitch_link']] = knee_pitch
        joint_angles[self.index_dof['left_independent_pitch_link']] = -knee_pitch
        joint_angles[self.index_dof['left_shin_pitch_mimic_link']] = knee_pitch
        joint_angles[self.index_dof['left_ankle_pitch_link']] = l_pitch
        joint_angles[self.index_dof['left_ankle_roll_link']] = l_roll - waist_roll

        r_x, r_y, r_z, r_roll, r_pitch, r_yaw = right_foot
        r_x -= self.OFFSET_X
        r_y += self.OFFSET_W
        r_z = self.L1 + self.L12 + self.L2 + self.L3 - r_z
        r_x2 = r_x * math.cos(r_yaw) + r_y * math.sin(r_yaw)
        r_y2 = -r_x * math.sin(r_yaw) + r_y * math.cos(r_yaw)
        r_z2 = r_z - self.L3
        waist_roll = math.atan2(r_y2, r_z2)
        r2 = r_y2**2 + r_z2**2
        r_z3 = math.sqrt(max(r2 - r_x2**2, 0.0)) - self.L12
        pitch = math.atan2(r_x2, r_z3)
        l = math.sqrt(r_x2**2 + r_z3**2)
        knee_disp = math.acos(min(max(l/(2.0*self.L1), -1.0), 1.0))
        waist_pitch = - pitch - knee_disp
        knee_pitch = - pitch + knee_disp
        joint_angles[self.index_dof['right_waist_yaw_link']] = r_yaw
        joint_angles[self.index_dof['right_waist_roll_link']] = waist_roll
        joint_angles[self.index_dof['right_waist_pitch_link']] = waist_pitch
        joint_angles[self.index_dof['right_knee_pitch_link']] = -waist_pitch
        joint_angles[self.index_dof['right_waist_pitch_mimic_link']] = waist_pitch
        joint_angles[self.index_dof['right_shin_pitch_link']] = knee_pitch
        joint_angles[self.index_dof['right_independent_pitch_link']] = -knee_pitch
        joint_angles[self.index_dof['right_shin_pitch_mimic_link']] = knee_pitch
        joint_angles[self.index_dof['right_ankle_pitch_link']] = r_pitch
        joint_angles[self.index_dof['right_ankle_roll_link']] = r_roll - waist_roll

        return joint_angles
