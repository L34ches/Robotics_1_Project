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
    Nmax = 5000
    alpha = 0.5
    q0 = np.array([115, 50, 75, 30, 30]) * np.pi / 180
    Pd = np.array([[0.2058], [0.1188], [0.1464]])
    q0 = np.array([90, 90, 81, 90, 89]) * np.pi / 180
    Pd = np.reshape([0.384, 0.046, 0], (-1, 1))
    #Pd = np.array([[0.2058], [0.1188]])
    q0s = np.array([[89.34519109,  89.50889332 , 79.19565301 , 89.50889332,  90.98221336],
                    [89.4270422 ,   43.83597194 , 134.03589911 ,  45.22744087 , 136.49143252],
                    [28.52981369 ,  34.91420055 , 149.91501515 , 159.73714878 ,  80.25971748],
                    [129.28853452 , 120.36676314  , 50.38406102 , 120.77601871 , 126.26004332],
                    [90.40925557 , 43.34486525,  34.50494499 , 66.34502817,  16.66140222]])*np.pi/180
    Pds = np.array([[0.046, 0.0, 0.384], [0.169, 0.0, 0.327], [-0.02, 0.047, 0.267], [-0.029, -0.045, 0.389], [0.231, 0, 0.081]])
    #q0 = np.array([89.4270422 ,   43.83597194 , 134.03589911 ,  45.22744087 , 136.49143252])*np.pi/180
    #Pd = np.array([[0.169], [0.0], [0.327]])
    #q0 = np.array([28.52981369 ,  34.91420055 , 149.91501515 , 159.73714878 ,  80.25971748])*np.pi/180
    #Pd = np.array([[-0.02], [0.047], [0.267]])
    #q0 = np.array([129.28853452 , 120.36676314  , 50.38406102 , 120.77601871 , 126.26004332])*np.pi/180
    #Pd = np.array([[-0.029], [-0.045], [0.389]])
    #q0 = np.array([90.40925557 , 43.34486525,  34.50494499 , 66.34502817,  16.66140222])*np.pi/180
    #Pd = np.array([[0.081], [0], [0.231]])
    '''
    res = np.zeros((5, 1))
    for i in range(5):
        q_approx = IK_Jb(q0s[i, :], np.reshape(Pds[i, :], (-1, 1)), Nmax, alpha, tol)
        res = np.hstack((res, np.reshape(q_approx[:, -1] * 180 / np.pi, (-1, 1))))
    res = res[:, 1:]
    '''
    
    Pd = np.array([[0.21], [-0.107], [0.05]])
    q_approx = IK_Jb(q0, Pd, Nmax, alpha, tol)
