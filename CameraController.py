# import statements
import cv2
import numpy as np


class CameraController:
    def __init__(self):
        # Camera Position and Rotation compared to end effector
        # TODO: Identify PTC and K for Dofbot Camera
        self.RTC = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        self.PTC = None
        self.K = np.array([[12, 0, 24], [0, 12, 24], [0, 0, 1]])    # Camera Calibration Matrix
        self.Kd = None   # Camera Distortion Matrix
        self.camera = None

    def cameraPose(self, R0T: np.ndarray, P0T: np.ndarray) -> tuple:
        """
        Get the pose of the camera relative to the base frame

        :param R0T: Rotation of the end effector
        :param P0T: Position of the end effector
        """
        R0C = np.matmul(R0T, self.RTC)
        P0C = P0T + np.matmul(R0T, self.PTC)
        return R0C, P0C

    def targetCameraPosition(self, PMi):
        """
        Finds the target object's position relative to the camera

        :param PMi: List of tuple points of the target object
        :returns: A tuple containing the target objects rotation and position
        """
        yi = [np.matmul(np.invert(self.K), np.array([PMi[i][0], PMi[i][1], 1], dtype=np.float)) \
              for i in range(len(PMi))]
        I = np.eye(3)
        C = np.zeros((3*len(PMi), 13))
        for i in range(len(PMi)):
            kron = np.kron(I, np.array([PMi[i][0], PMi[i][1], 1], dtype=np.float))
            # Set Kron product values
            C[3*i][0:9] = kron[0][:]
            C[3*i+1][0:9] = kron[1][:]
            C[3*i+2][0:9] = kron[2][:]
            # Set identity matrix values
            C[3*i][9:12] = I[0][:]
            C[3*i+1][9:12] = I[1][:]
            C[3*i+2][9:12] = I[2][:]
            # Set Y values
            C[3*i][12] = yi[i][0]
            C[3*i+1][12] = yi[i][1]
            C[3*i+2][12] = yi[i][2]
        # Compute SVD of C
        U, S, Vh = np.linalg.svd(C)
        # Solve Cx=0 using SVD results
        x = [np.transpose(Vh)[idx][12] for idx in range(np.size(np.transpose(Vh), 1))]
        # Extract RCM and RCM from x
        PCM = x[9:12]
        R1 = x[0:3]
        R2 = x[3:6]
        R3 = x[6:9]
        RCMtemp = np.array([R1, R2, R3])
        Ur, Sr, Vhr = np.linalg.svd(RCMtemp)
        Rh = np.matmul(Ur, Vhr)
        RCM = np.sign(np.linalg.det(Rh))*Rh
        w = x[9:]
        return RCM, PCM, w

    def targetBasePosition(self, RCM, PCM, R0T, P0T):
        """
        Gives the location of the target in the base frame

        :param RCM: Rotation of the object between the camera frame and the object
        :param PCM: Position of the object between the camera frame and the object
        """
        R0C, P0C = self.cameraPose(R0T, P0T)
        R0M = np.matmul(R0C, RCM)
        P0M = P0C + np.matmul(R0C, PCM)
        return R0M, P0M

    def takePicture(self):
        result, image = self.camera.read()
        return result, image

    def connectCamera(self):
        self.camera = cv2.VideoCapture(0)

    def identifyTarget(self, image):
        # Grayscale Image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Set grayscale Threshold
        _, threshold = cv2.threshold()
