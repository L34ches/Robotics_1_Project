# Import Statements
import Arm_Lib as armLib
import numpy as np
import time as time
import inverse_kinematics as ik
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


class Dofbot(RobotArm):
    """Class for the Dofbot Robotic Arm"""

    def __init__(self):
        """Constructor for the Dofbot Class"""
        super().__init__(nServos=5)  # Call to Super class Constructor
        self.limits = [180, 180, 180, 180, 270]  # Servo Rotation Limits for reading
        self._l = [0.061, 0.0435, 0.08285, 0.08285, 0.07385, 0.05457]  # Length of each segment
        self.ik_a = [0, 0, self._l[2], self._l[3], 0, 0]  # Values of "a" for inverse kinematics
        self.ik_alpha = [90, 0, 0, 90, 0]  # Values of "alpha" for inverse kinematics
        self.ik_d = [self._l[0]+self._l[1], 0, 0, 0, self._l[4]+self._l[5]]  # Values of "d" for inverse kinematics
        self.ik_theta = [0, 0, -90, 0, 0]  # Values of "theta" for inverse kinematics
        self.h = [ez, -ey, -ey, -ey, -ex]  # Values of h for Jacobian calculation

    def connect(self) -> bool:
        """
        Connect to the Dofbot arm

        :return: True if connected successfully, otherwise false
        """
        self.arm = armLib.Arm_Device()
        return True if self.arm is not None else False

    def setLED(self, r: int, g: int, b: int) -> None:
        """
        Sets the Brightness of the LED using an RGB value

        :param r: The brightness of the red, between 0 and 255
        :param g: The brightness of the green, between 0 and 255
        :param b: The brightness of the blue, between 0 and 255
        :return: None
        """
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255, "Brightness for LED is out of bounds"
        self.arm.Arm_RGB_set(r, g, b)

    def soundBuzzer(self, timeOn: int=255) -> None:
        """
        Sounds the buzzer.

        :param timeOn: The amount of time to sound the buzzer for in 100s of milliseconds, from 0-255
        :return: None
        """
        self.arm.Arm_Buzzer_On(timeOn)

    def setServoAngle(self, servoId: int, angle: float, moveTime: float) -> int:
        """
        Sets a servo to a given angle without PID control.

        :param servoId: The ID of the Servo to be moved
        :param angle: The angle to set the servo to
        :param moveTime: The amount of time in milliseconds to move the arm, set to 0 for fastest speed
        :return: Actual position of the servo in degrees
        """
        self.arm.Arm_serial_servo_write(servoId, angle, moveTime)  # Move the Arm
        time.sleep(moveTime / 1000)  # Wait for the Arm to finish Moving
        return self.arm.Arm_serial_servo_read(servoId)  # Return the angle the arm ended at

    def setAllServoAngles(self, angles: np.ndarray, moveTime: float) -> None:
        """
        Move all servos to the given angle

        :param angles: List of angles to move to. Should be same length as self.servos, and ordered from 1st servo to last
        :param moveTime: The amount of time in milliseconds to move the arm, set to 0 for fastest speed
        :return: None
        """
        assert len(angles) == len(self.servos), "Not enough Angles to Move Arm"  # Ensure enough angles were given
        self.arm.Arm_serial_servo_write6(*angles, time=moveTime)  # Begin to move the arm
        time.sleep(moveTime/1000)  # Wait for the arm to finish moving

    def readAllServoAngles(self) -> np.ndarray:
        """
        Reads the angles of all servos

        :return: An ndarray containing the angles for each servo in degrees from 1st servo to last
        """
        return np.array([self.arm.Arm_serial_servo_read(i) for i in self.servos])

    def getAnglesFromPosition(self, Rot, Pot) -> np.ndarray or bool:
        """
        Finds the Angles corresponding to Rot and Pot using Inverse Kinematics

        :param Rot: End Effect Rotation of the final Position
        :param Pot: End Effect Position of the final Position
        :return: An array of the angles with the same length as self.servos or False if no angles exist
        """
        q = np.empty((5, 4))
        # Use Inverse Kinematics to Find Joint Angles
        # Find theta using SP4
        k = -ey
        h = ez
        p = ex
        d = np.transpose(ez)*Rot*ex
        thetatmp = ik.subproblem4(h, p, k, d)
        if len(thetatmp) == 0:
            return False
        elif len(thetatmp) == 1:
            theta = np.array([thetatmp[0], None, thetatmp[0], None], dtype=float)
        else:
            theta = np.array([thetatmp[0], thetatmp[1], thetatmp[0], thetatmp[1]], dtype=float)
        # Find q1 using SP1
        for i in range(4):
            if theta[i] is not None:
                k = -ey
                p1 = ik.rot(-ey, theta[i])*ex
                p2 = Rot*ex
                q[0][i] = ik.subproblem1(p1, k, p2)
        # Find q5 using SP1
        for i in range(4):
            if theta[i] is not None:
                k = ex
                p1 = ik.rot(ey, theta[i])*ez
                p2 = np.transpose(Rot)*ez
                q[4][i] = ik.subproblem1(p1, k, p2)
        # Find q3 using SP3
        for i in range(2):
            if theta[i] is not None:
                Pprime = ik.rot(ez, -q[0][i]) * (Pot - self.ik_d[0] * ez)+ik.rot(ey, -theta[i])*self.ik_d[4]*ex
                k = -ey
                p1 = self.ik_a[3]*ez
                p2 = self.ik_a[2]*ex
                d = np.linalg.norm(Pprime)
                q3tmp = ik.subproblem3(p1, p2, k, d)
                if len(q3tmp) == 0:
                    return False
                if len(q3tmp) == 1:
                    q[2][i] = q3tmp
                else:
                    q[2][i] = q3tmp[0]
                    q[2][i+2] = q3tmp[1]
        # Find q2 using SP1
        for i in range(4):
            if q[2][i] is not None:
                Pprime = ik.rot(ez, -q[0][i]) * (Pot - self.ik_d[0] * ez) + ik.rot(ey, -theta[i]) * self.ik_d[4] * ex
                k = -ey
                p1 = (self.ik_a[2]*ex+ik.rot(ey, -q[2][i])*self.ik_a[3]*ez)
                p2 = Pprime
                q[1][i] = ik.subproblem1(p1, k, p2)
        # Find q4 using SP1
        for i in range(4):
            if q[2][i] is not None:
                q[3][i] = theta[i]-q[1][i]-q[2][i]
        # TODO Remove unnecessary values of q
        # TODO Convert q to degrees
        # TODO Verify that the Joint Angles exist and are within the capabilities of Dofbot
        # TODO Return the Angles

    def getPositionFromAngles(self, angles: np.ndarray) -> tuple:
        """
        Finds the end effect position given input angles for the arm using forward kinematics

        :param angles: An array of angles with length self.servos corresponding to each servo's position
        :return: An array of length 2 containing [Rot, Pot], False is the position is impossible
        """
        # Verify the given angles
        assert len(angles) == len(self.servos), "Number of angles does not match number of servos"
        if False in [0 <= angles[i] <= self.limits[i] for i in range(len(angles))]:  # Check if each angle is possible
            return False, False
        # Convert Degrees to Radians
        angles = angles * np.pi / 180
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
        Rot = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.dtype(float))
        for mat in Ri_i_1:
            Rot *= mat
        Pot = np.matrix([[0], [0], [0]], np.dtype(float))
        for i in range(len(Pi_i_1)):
            Pot += Pi_i_1[-1 - i]
            if i != len(Pi_i_1) - 1:
                Pot = Ri_i_1[-2 - i] * Pot
        # Return the position
        return Rot, Pot

    def setServoAnglePID(self, servoId: int, angle: float, moveTime: float) -> float:
        """
        Sets the angle of a servo using PID control

        :param servoId: The ID of the Servo to be moved
        :param angle: The angle to set the servo to
        :param moveTime: The amount of time in milliseconds to move the arm, set to 0 for fastest speed
        :return: The error between the desired angle and the final angle of the servo
        """
        # TODO Implement setServoAnglePID
        # PID Loop to set position of the arm
        # Determine the final error of the arm position
        # Return the error

    def setAllServoAnglesVerified(self, angles: np.ndarray, moveTime: float) -> bool:
        """
        Sets the angles of all servos after verifying that the final position is valid
        :param angles: An array of the angles being set
        :return: True if the movement is valid, false if otherwise
        """
        Rot, Pot = self.getPositionFromAngles(angles)
        if not Rot and Pot:
            return False
        # TODO Verify that the final position is valid
        self.setAllServoAngles(angles, moveTime)
        time.sleep(moveTime/1000)
        return True

    def jacobian(self, angles: np.ndarray) -> np.matrix:
        """
        Calculates the arm Jacobian for the given angles
        :param angles: The angles of the arm position
        :return: The arm Jacobian for the given input angles
        """
        # Rotations in reference to base frame
        Ri_i_1 = [np.ndarray((3, 3))] * 6
        Ri_i_1[0] = Rotz(angles[0])
        Ri_i_1[1] = Roty(-angles[1])
        Ri_i_1[2] = Roty(-angles[2])
        Ri_i_1[3] = Roty(-angles[3])
        Ri_i_1[4] = Rotx(-angles[4])
        Ri_i_1[5] = np.identity(3)
        # Displacements
        Pi_i_1 = [np.matrix([[0], [0], [0]], np.dtype(float))] * 6
        Pi_i_1[0] = (self._l[0] + self._l[1]) * ez
        Pi_i_1[2] = self._l[2] * ex
        Pi_i_1[3] = -self._l[3] * ez
        Pi_i_1[5] = -(self._l[4] + self._l[5]) * ex
        # Base Frame Reference
        R0i = [np.eye(3)] * 5
        for i in range(len(R0i)):
            for j in range(i+1):
               R0i[i] = np.matmul(R0i[i], Ri_i_1[j])
        P0i = [np.matrix([[0], [0], [0]], np.dtype(float))] * 6
        P0i[0] = Pi_i_1[0]
        for i in range(len(P0i)-1):
            P0i[i+1] = P0i[i]+R0i[i]*Pi_i_1[i+1]
        h0 = [np.matmul(R0i[i], self.h[i]) for i in range(len(self.h))]
        P0T = P0i[-1]
        cross = [np.cross(np.transpose(h0[i]), np.transpose(P0T-P0i[i])) for i in range(len(h0))]
        J = np.matrix(np.zeros((6, 5)))
        for i in range(5):
            J[:,i] = np.vstack((h0[i], np.transpose(cross[i])))
        return J
