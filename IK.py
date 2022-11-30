import numpy as np
import subproblems as sp
import FK

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

L1 = l0+l1
L4 = l4+l5

# Set up the position vectors between subsequent frames
P01 = (l0+l1)*ez        # translation between base frame and 1 frame in base frame
P12 = np.zeros((3,1))   # translation between 1 and 2 frame in 1 frame
P23 = -l2*ey             # translation between 2 and 3 frame in 2 frame
P34 = -l3*ez            # translation between 3 and 4 frame in 3 frame
P45 = np.zeros((3,1))   # translation between 4 and 5 frame in 4 frame
P5T = (l4+l5)*ey       # translation between 5 and tool frame in 5 frame

# inverse kinematics using subproblems
# returns all possible combinations of q1 ~ q5
def invkin_subproblems_Dofbot(Rot,Pot):
    # Set up placeholder for possible q sets
    q = np.NaN * np.ones((5,4))

    # Get theta = q2+q3+q4 from Rot
    k = -ex;
    h = ez;
    p = ey;
    d = np.matmul(np.transpose(ez), np.matmul(Rot, ey));
    thetatmp = sp.subproblem4(h, p, k, d);
    if thetatmp[0] == thetatmp[1]:
        theta = np.array([thetatmp[0], np.NaN, thetatmp[0], np.NaN])
    else:
        theta = np.array([thetatmp[0], thetatmp[1], thetatmp[0], thetatmp[1]])

    # Get q1 from Rot
    for ii in range(4):
        if not np.isnan(theta[ii]):
            k = ez
            p1 = np.matmul(sp.rot(ex, -theta[ii]), ey)
            p2 = np.matmul(Rot, ey)
            q[0][ii] = sp.subproblem1(p1, p2, k)

    # Get q5 from Rot
    for ii in range(4):
        if not np.isnan(theta[ii]):
            k = -ey
            p1 = np.matmul(sp.rot(ex, theta[ii]), ez)
            p2 = np.matmul(np.transpose(Rot), ez)
            q[4][ii] = sp.subproblem1(p1, p2, k)

    # Get q3 from Pot
    for ii in range(2):
        if not np.isnan(theta[ii]):
            Pprime = np.matmul(sp.rot(ez,-q[0][ii]), (Pot-L1 * ez)) - np.matmul(sp.rot(ex,-theta[ii]), L4 * ey)
            k = -ex
            p1 = l3 * ez
            p2 = l2 * -ey
            d = np.linalg.norm(Pprime)
            q3tmp  = sp.subproblem3(p1, p2, k, d)
            if len(q3tmp) == 1:
                q[2][ii] = q3tmp[0]
            else:
                q[2][ii] = q3tmp[0]
                q[2][ii+2] = q3tmp[1]

    # Get q2 from Pot
    for ii in range(4):
        if not np.isnan(q[2][ii]):
            Pprime = np.matmul(sp.rot(ez,-q[0][ii]), (Pot-L1 * ez)) - np.matmul(sp.rot(ex,-theta[ii]), L4 * ey)
            k = -ex;
            p1 = -l2 * ey - l3 * np.matmul(sp.rot(ex, -q[2][ii]), ez)
            p2 = Pprime;
            q[1][ii] = sp.subproblem1(p1, p2, k);

    # Get q4 from theta
    for ii in range(4):
        if not np.isnan(q[2][ii]):
            q[3][ii] = theta[ii]-q[1][ii]-q[2][ii]
    
    # Remove nonvalid columns of q
    tmp = np.array([]).reshape(5, 0)
    for ii in range(3,-1,-1):
        if not sum(np.isnan(q[:, ii])):
            tmp = np.concatenate((tmp, q[:, ii].reshape(5, 1)), axis = 1)
    q = tmp
    
    # Convert q to degrees
    q = q * 180 / np.pi;
    
    for ii in range(np.size(q, 0)):
        for jj in range(np.size(q, 1)):
            if q[ii][jj] < -180:
                q[ii][jj] = q[ii][jj] + 360
            elif q[ii][jj] > 360:
                q[ii][jj] = q[ii][jj] - 360
    
    return q

if __name__ == "__main__":
    q = [0, 45, 135, 45, 135]
    ROT, POT = FK.fwkin_POE_Dofbot(q)
    res = invkin_subproblems_Dofbot(ROT, POT)
    for i in range(np.size(res, 1)):
        print("Sol {}:".format(i + 1), np.round(res[:, i], 4))