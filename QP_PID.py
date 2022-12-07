import numpy as np
import FK
import cvxopt
import matplotlib.pyplot as plt

qlimit = np.hstack((np.zeros((5, 1)), np.array([[180], [180], [180], [180], [270]]) * np.pi / 180))

def cvxopt_solve_qp(P, q, G, h, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1]))

# returns [q_lambda,lambda,P0T_lambda,R0T_lambda]
def qpPathGen_positionOnly(q0, P0Td, epsilon_p, N, Kp, Ki, Kd):
    n = len(q0) # number of joints
    q_prime_min = -np.inf * np.ones((n, 1))
    q_prime_max = np.inf * np.ones((n,1))
    lamb = [i / N for i in range(N + 1)]
    
    R0T0, P0T0 = FK.fwkin_POE_Dofbot(q0)

    # Compute Path in Task Space
    Pdes_lambda = np.zeros((3, N + 1))
    dP0T_dlambda = P0Td - P0T0 # constant in lamb
    for k in range(N + 1):
        Pdes_lambda[:, k] = np.transpose((1-lamb[k]) * P0T0 + lamb[k] * P0Td)
    
    # Solve QP Problem and Generate Joint Space Path
    q_prime = np.zeros((n, N + 1))
    q_lambda = np.zeros((n, N + 2))
    q_lambda[:, 0] = q0
    P0T_lambda = np.zeros((3, N + 2))
    R0T_lambda = np.zeros((3, 3, N + 2))
    P0T_lambda[:, 0] = np.transpose(P0T0)
    R0T_lambda[:, :, 0] = R0T0
    qprev = q0
    e_i = 0
    e_p = 0
    for k in range(N + 1):
        lb, ub = qprimelimits_full(qlimit,qprev,N,q_prime_max,q_prime_min)
        J = FK.jacobian(qprev)
        e = np.reshape(Pdes_lambda[:, k] - P0T_lambda[:, -1], (-1, 1))
        e_i += e / N
        e_d = (e - e_p) * N
        e_p = e
        u_pid = Kp * e + Ki * e_i + Kd * e_d
        vt = dP0T_dlambda + u_pid
        # Hint: We only need the position rows of the Jacobian. MATLAB index starts at 1.
        # To get rows i through j (including j) we use J(i:j,:)
        H = getqp_H_positionOnly(qprev, J[3:, :], vt, epsilon_p)        
        f = getqp_f_positionOnly(qprev, epsilon_p)
        A = np.vstack((-np.eye(n + 1), np.eye(n + 1)))
        b = np.vstack((-lb, ub))
        q_prime_temp = cvxopt_solve_qp(H, f, A, b)
        q_prime_temp = q_prime_temp[:n]
        q_prime[:, k] = q_prime_temp
        qprev = qprev + (1 / N) * q_prime_temp
        q_lambda[:, k+1] = qprev
        Rtemp, Ptemp = FK.fwkin_POE_Dofbot(qprev)
        P0T_lambda[:, k+1] = np.transpose(Ptemp)
        R0T_lambda[:, :, k+1] = Rtemp
    # Chop off excess
    q_lambda = q_lambda[:, :-1]
    P0T_lambda = P0T_lambda[:, :-1]
    R0T_lambda = R0T_lambda[:, :, :-1]
    return q_lambda, lamb, P0T_lambda, R0T_lambda, Pdes_lambda

# returns [lb,ub]
def qprimelimits_full(qlimit, qprev, N, qpmax, qpmin):
    n = len(qlimit)
    # Compute limits due to joint stops
    lb_js = N * (qlimit[:, 0] - np.transpose(qprev))
    ub_js = N * (qlimit[:, 1] - np.transpose(qprev))
    # Compare and find most restrictive bound
    lb = np.zeros((n + 1, 1))
    ub = np.zeros((n + 1, 1))
    ub[-1] = 1
    for k in range(n):
        if lb_js[k] > qpmin[k]:
            lb[k] = lb_js[k]
        else:
            lb[k] = qpmin[k]
             
        if ub_js[k] < qpmax[k]:
            ub[k] = ub_js[k]
        else:
            ub[k] = qpmax[k]
    return lb, ub

# return [ H ]
def getqp_H_positionOnly(dq, J, vp, ep):
    n = len(dq)
    Jm = np.hstack((J, np.zeros((3, 1))))
    Vm = np.hstack((np.zeros((3, n)), vp))
    Em = np.hstack((np.zeros((1, n)), np.reshape([np.sqrt(ep)], (1, 1))))
    H1 = np.matmul(np.transpose(Jm), Jm)
    H2 = np.vstack((np.zeros((3, n + 1)), Vm))
    H2 = np.matmul(np.transpose(H2), H2)
    H3 = -2 * np.matmul(np.transpose(Jm), Vm)
    H3 = (H3 + np.transpose(H3)) / 2
    H4 = np.vstack((np.zeros((1, n + 1)), Em))
    H4 = np.matmul(np.transpose(H4), H4)
    H = 2 * (H1 + H2 + H3 + H4)
    return H

# return [ f ]
def getqp_f_positionOnly(dq, ep):
    f = -2 * np.transpose(np.hstack((np.zeros((1, len(dq))), np.reshape([ep], (1, 1)))))
    return f

if __name__ == "__main__":
    q0 = np.array([135, 70, 20, 20, 0]) * np.pi / 180
    q0 = np.array([0, 45, 135, 45, 135])*np.pi/180
    #q0 = np.array([180, 90, 0, 0, 90])*np.pi/180
    R0, P0 = FK.fwkin_POE_Dofbot(q0)
    P0Td = P0 - np.array([[0.02], [-0.08], [0]])
    P0Td = P0 - np.array([[0.0], [0], [0.05]])
    #P0Td = P0 - np.array([[0.12], [0.1], [0]])
    N = 100
    epsilon_p = 0.1
    Kp = 0.063
    Ki = 0
    Kd = 0
    q_lambda, lamb, P0T_lambda, R0T_lambda, Pdes_lambda = qpPathGen_positionOnly(q0, P0Td, epsilon_p, N, Kp, Ki, Kd)
    #np.save('data.npy', q_lambda)
    ax = plt.axes(projection ='3d')
    ax.plot3D(P0T_lambda[0, :], P0T_lambda[1, :], P0T_lambda[2, :], ".")
    ax.plot3D(Pdes_lambda[0, :], Pdes_lambda[1, :], Pdes_lambda[2, :], "*")
    ax.set_xlim([-1, 1])
    #ax.set_ylim([-1, 1])
    #ax.set_zlim([-0.1, 0.1])
    #ax.set_xlim([0, 0.2])
    #ax.set_ylim([0.05, 0.25])
    print(Pdes_lambda[:, -1] - P0T_lambda[:, -1])