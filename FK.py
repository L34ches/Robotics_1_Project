import numpy as np
import subproblems as sp

# Define major axis
ex = np.array([[1],[0],[0]])
ey = np.array([[0],[1],[0]])
ez = np.array([[0],[0],[1]])

# Define the link lengths in meters
l0 = 0.061      # base to servo 1
l1 = 0.0435     # servo 1 to servo 2
l2 = 0.08285    # servo 2 to servo 3
l3 = 0.08285    # servo 3 to servo 4
l4 = 0.07385    # servo 4 to servo 5
l5 = 0.05457    # servo 5 to gripper 

# return a rotation matrix for a radians q rotation around x-axis
def Rx(q):
    return np.array([[1, 0, 0], 
                     [0, np.cos(q), -np.sin(q)], 
                     [0, np.sin(q), np.cos(q)]])

# return a rotation matrix for a radians q rotation around y-axis
def Ry(q):
    return np.array([[np.cos(q), 0, np.sin(q)], 
                     [0, 1, 0], 
                     [-np.sin(q), 0, np.cos(q)]])

# return a rotation matrix for a radians q rotation around z-axis
def Rz(q):
    return np.array([[np.cos(q), -np.sin(q), 0], 
                     [np.sin(q), np.cos(q), 0], 
                     [0, 0, 1]])

# convert from degrees to radians
def degToRad(q):
    return q * np.pi / 180

# R0T = R01 * R12 * R23 * R34 * R45
# q1, ..., q5 in degrees
def R0T(q1, q2, q3, q4, q5):
    q1 = degToRad(q1)
    q2 = degToRad(q2)
    q3 = degToRad(q3)
    q4 = degToRad(q4)
    q5 = degToRad(q5)
    return np.matmul(np.matmul(sp.rot(ez, q1), sp.rot(ey, -q2 - q3 - q4)), sp.rot(ex, -q5))

# P12 = P45 = 0
# P0T = P01 + R02 * P23 + R03 * P34 + R05 * P5T
# q1, ..., q5 in degrees
def P0T(q1, q2, q3, q4, q5):
    q1 = degToRad(q1)
    q2 = degToRad(q2)
    q3 = degToRad(q3)
    q4 = degToRad(q4)
    q5 = degToRad(q5)
    R02 = np.matmul(sp.rot(ez, q1), sp.rot(ey, -q2))
    R03 = np.matmul(R02, sp.rot(ey, -q3))
    R05 = np.matmul(np.matmul(R03, sp.rot(ey, -q4)), sp.rot(ex, -q5))
    return (l0 + l1) * ez + l2 * np.matmul(R02, ex) - l3 * np.matmul(R03, ez) - (l4 + l5) * np.matmul(R05, ex)

# forward kinematics using POE method
# returns R_0T and P_0T
def fwkin_POE_Dofbot(q):
    ROT = R0T(q[0], q[1], q[2], q[3], q[4])
    POT = P0T(q[0], q[1], q[2], q[3], q[4])
    return ROT, POT

if __name__ == "__main__":
    q = [45, 45, 45, 45, 45]
    ROT, POT = FK.fwkin_POE_Dofbot(q)