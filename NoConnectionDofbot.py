# Import Statements
import numpy as np
from rotation import ex, ey, ez, Rotx, Roty, Rotz


class RobotArm:
    """Parent Class for Generic Robot Arm"""

    def __init__(self, nServos: int):
        """
        Constructor for the Robot Arm Class
        :param nServos: Number of Servos the arm has to control
        """
        self.arm = None
        self.servos = np.linspace(1, nServos, nServos)

    def connect(self) -> bool:
        """
        Connects the robot arm software to the hardware.
        :return: True if connected successfully, false otherwise
        """
        return False


class NoConnectDofbot(RobotArm):
    """A Clone of the Dofbot Class with all connections removed to allow for debugging"""

    def __init__(self):
        """Constructor for the NoConnectDofbot Class"""
        super().__init__(nServos=5)  # Call to Super class Constructor
        self.limits = [180, 180, 180, 180, 270]  # Servo Rotation Limits for reading
        self._l = [0.061, 0.0435, 0.08285, 0.08285, 0.07385, 0.05457]  # Length of each segment
        self.ik_a = [0, 0, self._l[2], self._l[3], 0, 0]  # Values of "a" for inverse kinematics
        self.ik_alpha = [90, 0, 0, 90, 0]  # Values of "alpha" for inverse kinematics
        self.ik_d = [self._l[0] + self._l[1], 0, 0, 0, self._l[4] + self._l[5]]  # Values of "d" for inverse kinematics
        self.ik_theta = [0, 0, -90, 0, 0]  # Values of "theta" for inverse kinematics

    def getPositionFromAngles(self, angles: np.ndarray) -> tuple or False:
        """
        Finds the end effect position given input angles for the arm

        :param angles: An array of angles with length self.servos corresponding to each servo's position
        :return: An array of length 2 containing [Rot, Pot], False is the position is impossible
        """
        # Verify the given angles
        assert len(angles) == len(self.servos), "Number of angles does not match number of servos"
        if False in [0 <= angles[i] <= self.limits[i] for i in range(len(angles))]:  # Check if each angle is possoble
            return False
        # Convert Degrees to Radians
        angles = angles * np.pi/180
        # Use Forward Kinematics to find the end effect position
        # Find R_{i, i+1}
        Ri_i_1 = [np.ndarray((3, 3))] * 6
        Ri_i_1[0] = Rotz(angles[0])
        Ri_i_1[1] = Roty(-angles[1])
        Ri_i_1[2] = Roty(-angles[2])
        Ri_i_1[3] = Roty(-angles[3])
        Ri_i_1[4] = Rotx(-angles[4])
        Ri_i_1[5] = np.identity(3)
        # Find P_{i-1, i}
        Pi_i_1 = [np.matrix([[0], [0], [0]], np.dtype(float))] * 6
        Pi_i_1[0] = (self._l[0] + self._l[1]) * ez
        Pi_i_1[2] = self._l[2] * ex
        Pi_i_1[3] = -self._l[3] * ez
        Pi_i_1[5] = -(self._l[4] + self._l[5]) * ex
        # Calculate Rot and Pot
        Rot =np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.dtype(float))
        for mat in Ri_i_1:
            Rot *= mat
        Pot = np.matrix([[0], [0], [0]], np.dtype(float))
        for i in range(len(Pi_i_1)):
            Pot += Pi_i_1[-1 - i]
            if i != len(Pi_i_1)-1:
                Pot = Ri_i_1[-2 - i] * Pot
        # Return the position
        return Rot, Pot
