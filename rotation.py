import numpy as np

ex = np.matrix([[1], [0], [0]], np.dtype(float))
ey = np.matrix([[0], [1], [0]], np.dtype(float))
ez = np.matrix([[0], [0], [1]], np.dtype(float))


def Rotx(angle: float) -> np.ndarray:
    """
    Calculates the 3D Rotation Matrix about the X-axis
    :param angle: The angle of the rotation in radians
    :return: The 3D rotation matrix
    """
    return np.matrix([[1, 0, 0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle), np.cos(angle)]], np.dtype(float))


def Roty(angle: float) -> np.ndarray:
    """
    Calculates the 3D Rotation Matrix about the Y-axis
    :param angle: The angle of the rotation in radians
    :return: The 3D rotation matrix
    """
    return np.matrix([[np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]], np.dtype(float))


def Rotz(angle: float) -> np.ndarray:
    """
    Calculates the 3D Rotation Matrix about the Z-axis
    :param angle: The angle of the rotation in radians
    :return: The 3D rotation matrix
    """
    return np.matrix([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]], np.dtype(float))