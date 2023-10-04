from controller import Robot, Motor
import numpy as np
import sys


class BipedSim():

    def __init__(self, legL, legR, acc, gyro):
        self.legL = legL
        self.legR = legR
        self.acc = acc
        self.gyro = gyro
