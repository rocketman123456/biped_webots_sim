'''
***************************************************************************
Modern Robotics: Mechanics, Planning, and Control.
Code Library
***************************************************************************
Author: Huan Weng, Bill Hunt, Jarvis Schultz, Mikhail Todes,
Email: huanweng@u.northwestern.edu
Date: January 2018
***************************************************************************
Language: Python
Also available in: MATLAB, Mathematica
Required library: numpy
Optional library: matplotlib
***************************************************************************
'''

'''
*** IMPORTS ***
'''

import numpy as np
from modern_robotics.utils import *
from modern_robotics.ch03 import *

'''
*** CHAPTER 4: FORWARD KINEMATICS ***
'''


def FKinBody(M, Blist, thetalist):
    """Computes forward kinematics in the body frame for an open chain robot

    :param M: The home configuration (position and orientation) of the end-effector
    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-
             effector frame when the joints are at the specified coordinates
             (i.t.o Body Frame)

    Example Input:
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
    Output:
        np.array([[0, 1,  0,         -5],
                  [1, 0,  0,          4],
                  [0, 0, -1, 1.68584073],
                  [0, 0,  0,          1]])
    """
    T = np.array(M)
    for i in range(len(thetalist)):
        T = np.dot(T, MatrixExp6(VecTose3(np.array(Blist)[:, i] * thetalist[i])))
    return T


def FKinSpace(M, Slist, thetalist):
    """Computes forward kinematics in the space frame for an open chain robot

    :param M: The home configuration (position and orientation) of the end-
              effector
    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-
             effector frame when the joints are at the specified coordinates
             (i.t.o Space Frame)

    Example Input:
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        Slist = np.array([[0, 0,  1,  4, 0,    0],
                          [0, 0,  0,  0, 1,    0],
                          [0, 0, -1, -6, 0, -0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
    Output:
        np.array([[0, 1,  0,         -5],
                  [1, 0,  0,          4],
                  [0, 0, -1, 1.68584073],
                  [0, 0,  0,          1]])
    """
    T = np.array(M)
    for i in range(len(thetalist) - 1, -1, -1):
        T = np.dot(MatrixExp6(VecTose3(np.array(Slist)[:, i] * thetalist[i])), T)
    return T
