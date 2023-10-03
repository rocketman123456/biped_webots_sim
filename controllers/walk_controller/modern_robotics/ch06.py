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
from modern_robotics.ch04 import *
from modern_robotics.ch05 import *

'''
*** CHAPTER 6: INVERSE KINEMATICS ***
'''


def IKinBody(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) @ T))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    while err and i < maxiterations:
        thetalist = thetalist + np.linalg.pinv(JacobianBody(Blist, thetalist)) @ Vb
        i = i + 1
        Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) @ T))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    return (thetalist, not err)


def IKinSpace(Slist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the space frame for an open chain robot

    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Slist = np.array([[0, 0,  1,  4, 0,    0],
                          [0, 0,  0,  0, 1,    0],
                          [0, 0, -1, -6, 0, -0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([ 1.57073783,  2.99966384,  3.1415342 ]), True)
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Tsb = FKinSpace(M, Slist, thetalist)
    Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    while err and i < maxiterations:
        thetalist = thetalist + np.linalg.pinv(JacobianSpace(Slist, thetalist)) @ Vs
        i = i + 1
        Tsb = FKinSpace(M, Slist, thetalist)
        Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    return (thetalist, not err)


def IKinBodyPseudoInverse(Blist, M, T, thetalist0, eomg, ev):
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) @ T))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    while err and i < maxiterations:
        jacobian = JacobianBody(Blist, thetalist)
        jacobian_inv = jacobian.T @ np.linalg.pinv(jacobian @ jacobian.T)
        thetalist = thetalist + jacobian_inv @ Vb
        i = i + 1
        Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) @ T))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    return (thetalist, not err)


def IKinSpacePseudoInverse(Slist, M, T, thetalist0, eomg, ev):
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Tsb = FKinSpace(M, Slist, thetalist)
    Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    while err and i < maxiterations:
        jacobian = JacobianSpace(Slist, thetalist)
        jacobian_inv = jacobian.T @ np.linalg.pinv(jacobian @ jacobian.T)
        thetalist = thetalist + jacobian_inv @ Vs
        i = i + 1
        Tsb = FKinSpace(M, Slist, thetalist)
        Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    return (thetalist, not err)


def IKinBodyDamped(Blist, M, T, thetalist0, lamb, eomg, ev):
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) @ T))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    while err and i < maxiterations:
        thetalist = thetalist + lamb * np.linalg.pinv(JacobianBody(Blist, thetalist)) @ Vb
        i = i + 1
        Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) @ T))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    return (thetalist, not err)


def IKinSpaceDamped(Slist, M, T, thetalist0, lamb, eomg, ev):
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Tsb = FKinSpace(M, Slist, thetalist)
    Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    while err and i < maxiterations:
        thetalist = thetalist + lamb * np.linalg.pinv(JacobianSpace(Slist, thetalist)) @ Vs
        i = i + 1
        Tsb = FKinSpace(M, Slist, thetalist)
        Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    return (thetalist, not err)


def IKinBodyDampedPseudoInverse(Blist, M, T, thetalist0, lamb1, lamb2, eomg, ev):
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) @ T))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    while err and i < maxiterations:
        jacobian = JacobianBody(Blist, thetalist)
        jacobian_inv = jacobian.T @ np.linalg.pinv(jacobian @ jacobian.T + lamb1 * np.eye(np.size(jacobian, 0)))
        thetalist = thetalist + lamb2 * jacobian_inv @ Vb
        i = i + 1
        Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) @ T))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    return (thetalist, not err)


def IKinSpaceDampedPseudoInverse(Slist, M, T, thetalist0, lamb1, lamb2, eomg, ev):
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Tsb = FKinSpace(M, Slist, thetalist)
    Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    while err and i < maxiterations:
        jacobian = JacobianSpace(Slist, thetalist)
        jacobian_inv = jacobian.T @ np.linalg.pinv(jacobian @ jacobian.T + lamb1 * np.eye(np.size(jacobian, 0)))
        thetalist = thetalist + lamb2 * jacobian_inv @ Vs
        i = i + 1
        Tsb = FKinSpace(M, Slist, thetalist)
        Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    return (thetalist, not err)


def IKinBodyDampedLeastSquare1(Blist, M, T, thetalist0, lamb, W, eomg, ev):
    # W = np.eye(np.size(thetalist, 1)) + (w-1) @ d @ d.T
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) @ T))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    while err and i < maxiterations:
        J = JacobianBody(Blist, thetalist)
        JJT = J.T @ W @ J + lamb * np.eye(thetalist.size)
        thetalist = thetalist + np.linalg.pinv(JJT) @ J.T @ W @ Vb
        i = i + 1
        Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) @ T))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    return (thetalist, not err)


def IKinSpaceDampedLeastSquare1(Slist, M, T, thetalist0, lamb, W, eomg, ev):
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Tsb = FKinSpace(M, Slist, thetalist)
    Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    while err and i < maxiterations:
        J = JacobianSpace(Slist, thetalist)
        JJT = J.T @ W @ J + lamb * np.eye(thetalist.size)
        thetalist = thetalist + np.linalg.pinv(JJT) @ J.T @ W @ Vs
        i = i + 1
        Tsb = FKinSpace(M, Slist, thetalist)
        Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    return (thetalist, not err)

# method with error
def IKinBodyDampedLeastSquare2(Blist, M, T, thetalist0, lamb, W, dt, eomg, ev):
    # W = np.eye(np.size(thetalist, 1)) + (w-1) @ d @ d.T
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) @ T))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    while err and i < maxiterations:
        J = JacobianBody(Blist, thetalist)
        JJT = W @ J @ J.T @ W + lamb * np.eye(thetalist.size)
        thetalist = thetalist + dt * W @ J.T @ np.linalg.pinv(JJT) @ W @ Vb
        i = i + 1
        Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) @ T))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    return (thetalist, not err)

# method with error
def IKinSpaceDampedLeastSquare2(Slist, M, T, thetalist0, lamb, W, dt, eomg, ev):
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Tsb = FKinSpace(M, Slist, thetalist)
    Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    while err and i < maxiterations:
        J = JacobianSpace(Slist, thetalist)
        JJT = W @ J @ J.T @ W + lamb * np.eye(np.size(W, 0))
        thetalist = thetalist + dt * W @ J.T @ np.linalg.pinv(JJT) @ W @ Vs
        i = i + 1
        Tsb = FKinSpace(M, Slist, thetalist)
        Vs = Adjoint(Tsb) @ se3ToVec(MatrixLog6(TransInv(Tsb) @ T))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    return (thetalist, not err)
