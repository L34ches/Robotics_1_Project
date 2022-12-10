import matplotlib.pyplot as plt
import numpy as np
import FK, IK_J

# generate object avoidance path using potential field in task space
def taskSpacePath(q0, P0Td, N):
    lamb = [i / N for i in range(N + 1)]
    R0T0, P0T0 = FK.fwkin_POE_Dofbot(q0)
    # Compute Path in Task Space
    Pdes_lambda = np.zeros((3, N + 1))
    dP0T_dlambda = P0Td - P0T0 # constant in lamb
    for k in range(N + 1):
        Pdes_lambda[:, k] = np.transpose((1-lamb[k]) * P0T0 + lamb[k] * P0Td)   
    return Pdes_lambda

# compute the joint angles to reach each path position using jacobian (transpose) IK
def jointSpacePath(tol, Nmax, alpha, q0, P0):
    tol=np.reshape([0.0001,0.0001, 0.0001], (-1, 1))
    Nmax = 5000
    alpha = 50
    q_lambda = np.copy(q0)
    Pest = np.copy(P0)
    for i in range(1, np.size(P, 1)):
        Pd = P[:, i]
        Pd = np.reshape(Pd, (-1, 1))
        q_approx = IK_J.IK_Jb(np.reshape(q_lambda[:, -1], (-1)), Pd, Nmax, alpha, tol)  
        sol = np.reshape(q_approx[:, -1], (-1, 1))
        q_lambda = np.hstack((q_lambda, sol))
        Pest = np.hstack((Pest, FK.P0T((q_lambda[0, -1]), q_lambda[1, -1], q_lambda[2, -1], q_lambda[3, -1], q_lambda[4, -1])))
    return q_lambda, Pest

if __name__ == "__main__":
    # set up start point, end point, and circular obstacle in task space
    q0 = np.array([135, 90, 20, 20, 270]) * np.pi / 180
    #q0 = np.array([90, 90, 0, 0, 270])*np.pi/180
    P0 = FK.P0T(q0[0], q0[1], q0[2], q0[3], q0[4])
    P0 = np.reshape(np.array(P0), (-1, 1))
    #Pd = P0[0:2] + np.array([[0.07], [0]])
    Pd = np.array([[0.12], [-0.08]])
    Pd = np.vstack((Pd, P0[2]))
    N = 10
    P = taskSpacePath(q0, Pd, N)
    
    fig, ax = plt.subplots()
    ax.plot(P[0,:], P[1,:], '.', label=r"Task space path (potential field)")
    ax.axis('equal')
    
    tol=np.reshape([0.0001,0.0001, 0.0001], (-1, 1))
    Nmax = 500
    alpha = 50
    q0 = np.array([[135], [90], [20], [20], [270]]) * np.pi / 180
    #q0 = np.array([[90], [90], [0], [0], [270]]) * np.pi / 180
    q_lambda, Pest = jointSpacePath(tol, Nmax, alpha, q0, P0)
    #np.save("vertical_jointpath.npy", q_lambda)
    #np.save("horizontal_jointpath.npy", q_lambda)
    
    # plot inverse kinematic path in task space
    ax.plot(Pest[0,:], Pest[1,:], "*", label=r"Task space end effector path (IK)")
    ax.legend(loc="upper right")
    ax.set_xlabel("X-axis position")
    ax.set_xlabel("Y-axis position")
    ax.set_title("Path generated after IK path")
    plt.show()
