import matplotlib.pyplot as plt
import numpy as np
import FK, IK_J

# measures the distance between the path position to surface of the circular obstacle
def distObj(P, c, r):
    return np.linalg.norm(P - c) - r

# compute attrative force
def attractGoal(P, Pd, k):
    return -k * (P - Pd)

# compute repulsive force
def repelObj(P, eta, rho0, c, r):
    d = distObj(P, c, r);
    if d > rho0:
        return np.array([[0], [0]])
    else:
        return -eta * (1 / d - 1 / rho0) * (-1 / (d * d)) * 2 * (P - c) / np.linalg.norm(P - c)

# generate object avoidance path using potential field in task space
def taskSpacePath(k, eta, rho, alpha, P0, Pd, c, r, n):
    i = 0
    P = np.zeros((2,n + 1))
    P[:,0] = np.transpose(P0[0:2])
    while np.linalg.norm(P[:,i] - Pd) > 1e-4:
        P[:,i + 1] = P[:, i] + np.transpose(alpha * (attractGoal(np.reshape(P[:, i], (-1, 1)), Pd, k) + repelObj(np.reshape(P[:,i], (-1, 1)), eta, rho0, c, r)))
        i += 1
        if i > n - 1:
            break
    P = P[:,:i+1]
    return P

# compute the joint angles to reach each path position using jacobian (transpose) IK
def jointSpacePath(tol, Nmax, alpha, q0, P0):
    tol=np.reshape([0.0001,0.0001, 0.0001], (-1, 1))
    Nmax = 5000
    alpha = 50
    q_lambda = np.copy(q0)
    Pest = np.copy(P0)
    for i in range(1, np.size(P, 1)):
        Pd = np.concatenate((P[:, i], P0[2]))
        Pd = np.reshape(Pd, (-1, 1))
        q_approx = IK_J.IK_Jb(np.reshape(q_lambda[:, -1], (-1)), Pd, Nmax, alpha, tol)  
        sol = np.reshape(q_approx[:, -1], (-1, 1))
        q_lambda = np.hstack((q_lambda, sol))
        Pest = np.hstack((Pest, FK.P0T((q_lambda[0, -1]), q_lambda[1, -1], q_lambda[2, -1], q_lambda[3, -1], q_lambda[4, -1])))
    return q_lambda, Pest

if __name__ == "__main__":
    # set up start point, end point, and circular obstacle in task space
    q0 = np.array([135, 90, 20, 0, 90]) * np.pi / 180
    P0 = FK.P0T(q0[0], q0[1], q0[2], q0[3], q0[4])
    P0 = np.reshape(np.array(P0), (-1, 1))
    Pd = P0[0:2] + np.array([[0.01], [-0.15]])
    c = np.array([P0[0] + 0.01, [0.01]])
    
    #q0 = np.array([90, 135, 0, 0, 180]) * np.pi / 180
    #P0 = FK.P0T(q0[0], q0[1], q0[2], q0[3], q0[4])
    #P0 = np.reshape(np.array(P0), (-1, 1))
    #Pd = P0[0:2] + np.array([[0.08], [0]])
    #c = np.array([P0[0] + 0.03, [-0.01]])
    r = 0.015 * np.sqrt(2)
    n = 100
    
    # set up hyper-parameters
    k = 15
    k = 20
    eta = 0.0000002
    rho0 = 0.01
    alpha = 0.001
    P = taskSpacePath(k, eta, rho0, alpha, P0, Pd, c, r, n)
        
    # plot object avoidance path in task space
    obj = plt.Circle((c[0] , c[1]), r, fill = False)
    buf = plt.Circle((c[0] , c[1]), r + rho0, fill = False, linestyle='--')
    fig, ax = plt.subplots()
    ax.add_patch(obj)
    ax.add_patch(buf)
    ax.plot(P[0,:], P[1,:], label=r"Task space obstacle avoidance path (potential field)")
    ax.axis('equal')
    ax.set_xlabel("X-axis (meters)")
    ax.set_ylabel("Y-axis (meters)")
    ax.set_title("Obstacle avoidance path (potential field)")
    '''
    tol=np.reshape([0.0001,0.0001, 0.0001], (-1, 1))
    Nmax = 5000
    alpha = 50
    q0 = np.array([[135], [90], [20], [0], [90]]) * np.pi / 180
    #q0 = np.array([[90], [135], [0], [0], [180]]) * np.pi / 180
    q_lambda, Pest = jointSpacePath(tol, Nmax, alpha, q0, P0)
    #np.save("objavoidp.npy", q_lambda)
    
    # plot inverse kinematic path in task space
    ax.plot(Pest[0,:], Pest[1,:], "*", label=r"Task space end effector path (IK)")
    ax.axis('equal')
    ax.legend(loc="upper right")
    ax.set_xlabel("X-axis position")
    ax.set_ylabel("Y-axis position")
    #ax.set_title("Obstacle avoidance path")
    ax.set_title("Obstacle avoidance path (IK)")
    plt.show()
    '''