import math
import numpy as np
import warnings
import sys

if (sys.version_info > (3, 0)):
    xrange = range


def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat = np.zeros((3, 3))
    khat[0, 1] = -k[2]
    khat[0, 2] = k[1]
    khat[1, 0] = k[2]
    khat[1, 2] = -k[0]
    khat[2, 0] = -k[1]
    khat[2, 1] = k[0]
    return khat

def rot(k, theta):
    """
    Generates a 3 x 3 rotation matrix from a unit 3 x 1 unit vector axis
    and an angle in radians using the Euler-Rodrigues formula

        R = I + sin(theta)*hat(k) + (1 - cos(theta))*hat(k)^2

    :type    k: numpy.array
    :param   k: 3 x 1 unit vector axis
    :type    theta: number
    :param   theta: rotation about k in radians
    :rtype:  numpy.array
    :return: the 3 x 3 rotation matrix

    """
    I = np.identity(3)
    k = np.divide(k, np.linalg.norm(k))
    khat = hat(k)
    khat2 = np.matmul(khat, khat)
    return I + math.sin(theta) * khat + (1.0 - math.cos(theta)) * khat2

def subproblem0(p, q, k):
    """
    Solves canonical geometric subproblem 0, theta subtended between p and q according to

        q = rot(k, theta)*p
           ** assumes k'*p = 0 and k'*q = 0

    Requires that p and q are perpendicular to k. Use subproblem 1 if this is not
    guaranteed.
    :type    p: numpy.array
    :param   p: 3 x 1 vector before rotation
    :type    q: numpy.array
    :param   q: 3 x 1 vector after rotation
    :type    k: numpy.array
    :param   k: 3 x 1 rotation axis unit vector
    :rtype:  number
    :return: theta angle as scalar in radians
    """

    eps = np.finfo(np.float64).eps
    assert (np.matmul(np.transpose(k), p) < eps) and (np.matmul(np.transpose(k), q) < eps), \
        "k must be perpendicular to p and q"

    norm = np.linalg.norm

    ep = p / norm(p)
    eq = q / norm(q)

    theta = 2 * np.arctan2(norm(ep - eq), norm(ep + eq))

    if (np.matmul(np.transpose(k), np.cross(ep, eq, axis=0)) < 0):
        return -theta

    return theta


def subproblem1(p, q, k):
    """
    Solves canonical geometric subproblem 1, theta subtended between p and q according to

        q = rot(k, theta)*p

    :type    p: numpy.array
    :param   p: 3 x 1 vector before rotation
    :type    q: numpy.array
    :param   q: 3 x 1 vector after rotation
    :type    k: numpy.array
    :param   k: 3 x 1 rotation axis unit vector
    :rtype:  number
    :return: theta angle as scalar in radians
    """

    eps = np.finfo(np.float64).eps
    norm = np.linalg.norm

    if norm(np.subtract(p, q)) < np.sqrt(eps):
        return 0.0

    k = np.divide(k, norm(k))
    pp = np.subtract(p, np.matmul(np.transpose(p), k) * k)
    qp = np.subtract(q, np.matmul(np.transpose(q), k) * k)

    epp = np.divide(pp, norm(pp))
    eqp = np.divide(qp, norm(qp))

    theta = subproblem0(epp, eqp, k)

    if (np.abs(norm(p) - norm(q)) > norm(p) * 1e-2):
        warnings.warn("||p|| and ||q|| must be the same!!!")

    return theta


def subproblem2(p, q, k1, k2):
    """
    Solves canonical geometric subproblem 2, solve for two coincident, nonparallel
    axes rotation a link according to

        q = rot(k1, theta1) * rot(k2, theta2) * p

    solves by looking for the intersection between cones of

        rot(k1,-theta1)q = rot(k2, theta2) * p

    may have 0, 1, or 2 solutions


    :type    p: numpy.array
    :param   p: 3 x 1 vector before rotations
    :type    q: numpy.array
    :param   q: 3 x 1 vector after rotations
    :type    k1: numpy.array
    :param   k1: 3 x 1 rotation axis 1 unit vector
    :type    k2: numpy.array
    :param   k2: 3 x 1 rotation axis 2 unit vector
    :rtype:  list of number pairs
    :return: theta angles as list of number pairs in radians
    """

    eps = np.finfo(np.float64).eps
    norm = np.linalg.norm
    
    p = p / norm(p) * norm(q)
    k12 = np.matmul(np.transpose(k1), k2)
    pk = np.matmul(np.transpose(p), k2)
    qk = np.matmul(np.transpose(q), k1)

    # check if solution exists
    if (np.abs(1 - k12 ** 2) < eps):
        warnings.warn("No solution - k1 != k2")
        return []

    a = np.matmul(np.array([[1, -k12], [-k12, 1]]), np.array([[qk], [pk]])) / (1 - k12 ** 2)

    bb = (norm(q) ** 2 - norm(a) ** 2 - 2 * a[0][0] * a[1][0] * k12)
    if (np.abs(bb) < eps): bb = 0

    if (bb < 0):
        warnings.warn("No solution - no intersection found between cones")
        return []

    gamma = np.sqrt(bb) / norm(np.cross(k1, k2, axis=0))
    if (np.abs(bb) < eps):
        v = np.matmul(np.hstack((k1, k2)), a)
        theta2 = subproblem1(k2, p, v)
        theta1 = -subproblem1(k1, q, v)
        return [(theta1, theta2)]

    cm = np.hstack((k1, k2, np.cross(k1, k2, axis=0)))
    c1 = np.matmul(cm, np.vstack((a, gamma)))
    c2 = np.matmul(cm, np.vstack((a, -gamma)))
    theta1_1 = -subproblem1(q, c1, k1)
    theta1_2 = -subproblem1(q, c2, k1)
    theta2_1 = subproblem1(p, c1, k2)
    theta2_2 = subproblem1(p, c2, k2)
    return [(theta1_1, theta2_1), (theta1_2, theta2_2)]


def subproblem3(p, q, k, d):
    """
    Solves canonical geometric subproblem 3,solve for theta in
    an elbow joint according to

        || q + rot(k, theta)*p || = d

    may have 0, 1, or 2 solutions

    :type    p: numpy.array
    :param   p: 3 x 1 position vector of point p
    :type    q: numpy.array
    :param   q: 3 x 1 position vector of point q
    :type    k: numpy.array
    :param   k: 3 x 1 rotation axis for point p
    :type    d: number
    :param   d: desired distance between p and q after rotation
    :rtype:  list of numbers
    :return: list of valid theta angles in radians

    """

    norm = np.linalg.norm

    pp = np.subtract(p, np.matmul(np.transpose(k), p) * k)
    qp = np.subtract(q, np.matmul(np.transpose(k), q) * k)
    dpsq = d ** 2 - (np.matmul(np.transpose(k), np.subtract(p, q)) ** 2)

    bb = (norm(pp) ** 2 + norm(qp) ** 2 - dpsq) / (2 * norm(pp) * norm(qp))

    if dpsq < 0 or np.abs(bb) > 1:
        warnings.warn("No solution - no rotation can achieve specified distance")
        return []

    theta = subproblem1(pp / norm(pp), qp / norm(qp), k)

    phi = np.arccos(bb)
    if np.abs(phi) > 0:
        return [theta + phi, theta - phi]
    else:
        return [theta]


def subproblem4(p, q, k, d):
    """
    Solves canonical geometric subproblem 4, theta for static
    displacement from rotation axis according to

        d = p.T*rot(k, theta)*q

    may have 0, 1, or 2 solutions

    :type    p: numpy.array
    :param   p: 3 x 1 position vector of point p
    :type    q: numpy.array
    :param   q: 3 x 1 position vector of point q
    :type    k: numpy.array
    :param   k: 3x1 rotation axis for point p
    :type    d: number
    :param   d: desired displacement
    :rtype:  list of numbers
    :return: list of valid theta angles in radians
    """
    
    d = d / np.linalg.norm(p)
    p = p / np.linalg.norm(p)
    
    a = np.matmul(np.matmul(np.transpose(p), hat(k)), q)
    b = -np.matmul(np.transpose(p), np.matmul(np.matmul(hat(k), hat(k)), q))
    c = np.subtract(d, (np.matmul(np.transpose(p), q) - b))

    phi = np.arctan2(b, a)

    cond = c / np.linalg.norm([a, b])

    if abs(cond) > 1:
        return []

    psi = np.arcsin(cond)

    return [-phi + psi, -phi - psi + np.pi]