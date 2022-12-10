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

# Dofbot joint axis
h1 = ez
h2 = -ex
h3 = -ex
h4 = -ex
h5 = ey

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
# q1, ..., q5 in radians
def R0T(q1, q2, q3, q4, q5):
    return np.matmul(np.matmul(sp.rot(ez, q1), sp.rot(ex, -q2 - q3 - q4)), sp.rot(ey, q5))

# P12 = P45 = 0
# P0T = P01 + R02 * P23 + R03 * P34 + R05 * P5T
# q1, ..., q5 in radians
def P0T(q1, q2, q3, q4, q5):
    R02 = np.matmul(sp.rot(ez, q1), sp.rot(ex, -q2))
    R03 = np.matmul(R02, sp.rot(ex, -q3))
    R05 = np.matmul(np.matmul(R03, sp.rot(ex, -q4)), sp.rot(ey, q5))
    return (l0 + l1) * ez - l2 * np.matmul(R02, ey) - l3 * np.matmul(R03, ez) + (l4 + l5) * np.matmul(R05, ey)

# forward kinematics using POE method
# returns R_0T and P_0T
def fwkin_POE_Dofbot(q):
    q = np.reshape(q, (-1))
    ROT = R0T(q[0], q[1], q[2], q[3], q[4])
    POT = P0T(q[0], q[1], q[2], q[3], q[4])
    return ROT, POT

# q is a 5 * 1 vector of joint positions in radian
# return Dofbot jacobian matrix given joint positions
def jacobian(q):
    Pi_1_i = np.concatenate(((l0+l1)*ez, np.zeros((3,1)), -l2*ey, -l3*ez, np.zeros((3,1)), (l4+l5)*ey), 1)
    H = np.concatenate((h1, h2, h3, h4, h5), 1)
    H_0 = np.zeros((3,5))
    Roi = np.zeros((3,3,5))
    Ri_1_i = np.zeros((3,3,5))
    Poi = np.zeros((3,6))
    for ii in range(5):
        Ri_1_i[:,:,ii] = sp.rot(H[:,ii],q[ii])
        if ii == 0:
            Roi[:,:,ii] = Ri_1_i[:,:,ii]
            Poi[:,ii] = Pi_1_i[:,ii]
            Poi[:,ii+1] = Poi[:,ii] + np.matmul(Roi[:,:,ii], Pi_1_i[:,ii+1])
            H_0[:,ii] = np.matmul(Roi[:,:,ii], H[:,ii])
        else:
            Roi[:,:,ii] = np.matmul(Roi[:,:,ii-1], Ri_1_i[:,:,ii])
            Poi[:,ii+1] = Poi[:,ii] + np.matmul(Roi[:,:,ii], Pi_1_i[:,ii+1])
            H_0[:,ii] = np.matmul(Roi[:,:,ii], H[:,ii])
    Pot = Poi[:,5]
    J = np.zeros((6,5))
    for ii in range(5):
        J[:,ii] = np.concatenate((H_0[:,ii], np.cross(H_0[:,ii], np.subtract(Pot, Poi[:,ii]))))
    return J

if __name__ == "__main__":
    q = np.array([90, 45, 35, 65, 15]) * np.pi / 180
    q = np.array([57., 79., 36.,  0., 90.])*np.pi/180
    q = np.array([90, 45, 35, 65, 15])*np.pi/180
    ROT, POT = fwkin_POE_Dofbot(q)
    jac = jacobian(q)
    print(jac)