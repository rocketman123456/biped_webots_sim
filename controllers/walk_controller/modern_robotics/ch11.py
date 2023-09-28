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

'''
*** CHAPTER 11: ROBOT CONTROL ***
'''

def ComputedTorque(thetalist, dthetalist, eint, g, Mlist, Glist, Slist, \
                   thetalistd, dthetalistd, ddthetalistd, Kp, Ki, Kd):
    """Computes the joint control torques at a particular time instant

    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param eint: n-vector of the time-integral of joint errors
    :param g: Gravity vector g
    :param Mlist: List of link frames {i} relative to {i-1} at the home
                  position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :param thetalistd: n-vector of reference joint variables
    :param dthetalistd: n-vector of reference joint velocities
    :param ddthetalistd: n-vector of reference joint accelerations
    :param Kp: The feedback proportional gain (identical for each joint)
    :param Ki: The feedback integral gain (identical for each joint)
    :param Kd: The feedback derivative gain (identical for each joint)
    :return: The vector of joint forces/torques computed by the feedback
             linearizing controller at the current instant

    Example Input:
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        eint = np.array([0.2, 0.2, 0.2])
        g = np.array([0, 0, -9.8])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        thetalistd = np.array([1.0, 1.0, 1.0])
        dthetalistd = np.array([2, 1.2, 2])
        ddthetalistd = np.array([0.1, 0.1, 0.1])
        Kp = 1.3
        Ki = 1.2
        Kd = 1.1
    Output:
        np.array([133.00525246, -29.94223324, -3.03276856])
    """
    e = np.subtract(thetalistd, thetalist)
    return np.dot(MassMatrix(thetalist, Mlist, Glist, Slist), \
                  Kp * e + Ki * (np.array(eint) + e) \
                  + Kd * np.subtract(dthetalistd, dthetalist)) \
           + InverseDynamics(thetalist, dthetalist, ddthetalistd, g, \
                             [0, 0, 0, 0, 0, 0], Mlist, Glist, Slist)

def SimulateControl(thetalist, dthetalist, g, Ftipmat, Mlist, Glist, \
                    Slist, thetamatd, dthetamatd, ddthetamatd, gtilde, \
                    Mtildelist, Gtildelist, Kp, Ki, Kd, dt, intRes):
    """Simulates the computed torque controller over a given desired
    trajectory

    :param thetalist: n-vector of initial joint variables
    :param dthetalist: n-vector of initial joint velocities
    :param g: Actual gravity vector g
    :param Ftipmat: An N x 6 matrix of spatial forces applied by the end-
                    effector (If there are no tip forces the user should
                    input a zero and a zero matrix will be used)
    :param Mlist: Actual list of link frames i relative to i-1 at the home
                  position
    :param Glist: Actual spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :param thetamatd: An Nxn matrix of desired joint variables from the
                      reference trajectory
    :param dthetamatd: An Nxn matrix of desired joint velocities
    :param ddthetamatd: An Nxn matrix of desired joint accelerations
    :param gtilde: The gravity vector based on the model of the actual robot
                   (actual values given above)
    :param Mtildelist: The link frame locations based on the model of the
                       actual robot (actual values given above)
    :param Gtildelist: The link spatial inertias based on the model of the
                       actual robot (actual values given above)
    :param Kp: The feedback proportional gain (identical for each joint)
    :param Ki: The feedback integral gain (identical for each joint)
    :param Kd: The feedback derivative gain (identical for each joint)
    :param dt: The timestep between points on the reference trajectory
    :param intRes: Integration resolution is the number of times integration
                   (Euler) takes places between each time step. Must be an
                   integer value greater than or equal to 1
    :return taumat: An Nxn matrix of the controllers commanded joint forces/
                    torques, where each row of n forces/torques corresponds
                    to a single time instant
    :return thetamat: An Nxn matrix of actual joint angles
    The end of this function plots all the actual and desired joint angles
    using matplotlib and random libraries.

    Example Input:
        from __future__ import print_function
        import numpy as np
        from modern_robotics import JointTrajectory
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        # Initialize robot description (Example with 3 links)
        g = np.array([0, 0, -9.8])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        dt = 0.01
        # Create a trajectory to follow
        thetaend = np.array([np.pi / 2, np.pi, 1.5 * np.pi])
        Tf = 1
        N = int(1.0 * Tf / dt)
        method = 5
        traj = mr.JointTrajectory(thetalist, thetaend, Tf, N, method)
        thetamatd = np.array(traj).copy()
        dthetamatd = np.zeros((N, 3))
        ddthetamatd = np.zeros((N, 3))
        dt = Tf / (N - 1.0)
        for i in range(np.array(traj).shape[0] - 1):
            dthetamatd[i + 1, :] \
            = (thetamatd[i + 1, :] - thetamatd[i, :]) / dt
            ddthetamatd[i + 1, :] \
            = (dthetamatd[i + 1, :] - dthetamatd[i, :]) / dt
        # Possibly wrong robot description (Example with 3 links)
        gtilde = np.array([0.8, 0.2, -8.8])
        Mhat01 = np.array([[1, 0, 0,   0],
                           [0, 1, 0,   0],
                           [0, 0, 1, 0.1],
                           [0, 0, 0,   1]])
        Mhat12 = np.array([[ 0, 0, 1, 0.3],
                           [ 0, 1, 0, 0.2],
                           [-1, 0, 0,   0],
                           [ 0, 0, 0,   1]])
        Mhat23 = np.array([[1, 0, 0,    0],
                           [0, 1, 0, -0.2],
                           [0, 0, 1,  0.4],
                           [0, 0, 0,    1]])
        Mhat34 = np.array([[1, 0, 0,   0],
                           [0, 1, 0,   0],
                           [0, 0, 1, 0.2],
                           [0, 0, 0,   1]])
        Ghat1 = np.diag([0.1, 0.1, 0.1, 4, 4, 4])
        Ghat2 = np.diag([0.3, 0.3, 0.1, 9, 9, 9])
        Ghat3 = np.diag([0.1, 0.1, 0.1, 3, 3, 3])
        Gtildelist = np.array([Ghat1, Ghat2, Ghat3])
        Mtildelist = np.array([Mhat01, Mhat12, Mhat23, Mhat34])
        Ftipmat = np.ones((np.array(traj).shape[0], 6))
        Kp = 20
        Ki = 10
        Kd = 18
        intRes = 8
        taumat,thetamat \
        = mr.SimulateControl(thetalist, dthetalist, g, Ftipmat, Mlist, \
                             Glist, Slist, thetamatd, dthetamatd, \
                             ddthetamatd, gtilde, Mtildelist, Gtildelist, \
                             Kp, Ki, Kd, dt, intRes)
    """
    Ftipmat = np.array(Ftipmat).T
    thetamatd = np.array(thetamatd).T
    dthetamatd = np.array(dthetamatd).T
    ddthetamatd = np.array(ddthetamatd).T
    m,n = np.array(thetamatd).shape
    thetacurrent = np.array(thetalist).copy()
    dthetacurrent = np.array(dthetalist).copy()
    eint = np.zeros((m,1)).reshape(m,)
    taumat = np.zeros(np.array(thetamatd).shape)
    thetamat = np.zeros(np.array(thetamatd).shape)
    for i in range(n):
        taulist \
        = ComputedTorque(thetacurrent, dthetacurrent, eint, gtilde, \
                         Mtildelist, Gtildelist, Slist, thetamatd[:, i], \
                         dthetamatd[:, i], ddthetamatd[:, i], Kp, Ki, Kd)
        for j in range(intRes):
            ddthetalist = ForwardDynamics(thetacurrent, dthetacurrent, taulist, g, Ftipmat[:, i], Mlist, Glist, Slist)
            thetacurrent, dthetacurrent = EulerStep(thetacurrent, dthetacurrent, ddthetalist, 1.0 * dt / intRes)
        taumat[:, i] = taulist
        thetamat[:, i] = thetacurrent
        eint = np.add(eint, dt * np.subtract(thetamatd[:, i], thetacurrent))
    # Output using matplotlib to plot
    try:
        import matplotlib.pyplot as plt
    except:
        print('The result will not be plotted due to a lack of package matplotlib')
    else:
        links = np.array(thetamat).shape[0]
        N = np.array(thetamat).shape[1]
        Tf = N * dt
        timestamp = np.linspace(0, Tf, N)
        for i in range(links):
            col = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]
            plt.plot(timestamp, thetamat[i, :], "-", color=col, label = ("ActualTheta" + str(i + 1)))
            plt.plot(timestamp, thetamatd[i, :], ".", color=col, label = ("DesiredTheta" + str(i + 1)))
        plt.legend(loc = 'upper left')
        plt.xlabel("Time")
        plt.ylabel("Joint Angles")
        plt.title("Plot of Actual and Desired Joint Angles")
        plt.show()
    taumat = np.array(taumat).T
    thetamat = np.array(thetamat).T
    return (taumat, thetamat)
