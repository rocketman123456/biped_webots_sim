import numpy as np
import sys
from modern_robotics.utils import *
from modern_robotics.ch03 import *
from modern_robotics.ch04 import *

if __name__ == '__main__':
    print(NearZero(10))

    L0 = 0.05
    L1 = 0.15
    L2 = 0.15
    L3 = 0.05

    M = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, L0],
        [0, 0, 1, -L1-L2-L3],
        [0, 0, 0, 1]
    ])

    S1 = np.array([0, 0, 1, 0, 0, 0])
    S2 = np.array([0, -1, 0, 0, 0, -L1])
    S3 = np.array([1, 0, 0, 0, L2, 0])
    # S4 = np.array([0, 0, 0, 0, 0, 0])
    # S5 = np.array([0, 0, 0, 0, 0, 0])

    theta = np.array([0, 0, 0])

    T = FKinSpace(M, np.array([S1, S2, S3]).T, theta)
    print(T)
