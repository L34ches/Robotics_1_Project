from scipy.spatial.transform import Rotation
import numpy as np
import FK

# inverse kinematics (position only) using jacobian transpose
def IK_Jb(q0, Pd, Nmax, alpha, tol):
    n = len(q0) # number of joints
    q = np.zeros((n, Nmax+1))
    q[:, 0] = q0
    p0T = np.zeros((3, Nmax+1))
    iternum = 0
    P = FK.P0T(q0[0], q0[1], q0[2], q0[3], q0[4])
    dX = np.subtract(P[0:3, :], Pd)
    while not np.prod(abs(dX[0:3, :]) < tol):
        if iternum < Nmax:
            p0T[:, iternum] = np.reshape(FK.P0T(q[0][iternum], q[1][iternum], q[2][iternum], q[3][iternum], q[4][iternum]), (-1))
            Jq = FK.jacobian(q[:,iternum])
            dX = np.subtract(np.reshape(p0T[0:3,iternum], (-1, 1)), Pd)
            q[:, iternum + 1] = q[:, iternum] - alpha * np.reshape(np.matmul(np.transpose(Jq[3:6]), dX), (1, -1))
            iternum = iternum + 1
        else:
            break
    print(iternum)
    return q[:,0:iternum + 1]
    
if __name__ == "__main__":
    tol=np.reshape([0.0001,0.0001,0.0001], (-1, 1))
    Nmax = 100000
    alpha = 0.5
    q0 = np.array([115, 50, 75, 30, 30]) * np.pi / 180
    Pd = np.array([[0.2058], [0.1188], [0.1464]])
    #Pd = np.array([[0.2058], [0.1188]])
    q_approx = IK_Jb(q0, Pd, Nmax, alpha, tol)
