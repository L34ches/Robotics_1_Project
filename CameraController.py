# import statements
import cv2
import numpy as np


class CameraController:
    def __init__(self):
        # Camera Position and Rotation compared to end effector
        # TODO: Identify PTC for Dofbot Camera
        self.RTC = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        self.PTC = None
        self.K = np.array([[1020.4105, 0, 354.208471], [0, 034.79599, 238.13645], [0, 0, 1]])# Camera Calibration Matrix
        self.Kd = np.array([[-0.025676, -1.6639, -0.015317, -0.0051362, -39.96275]])   # Camera Distortion Matrix
        # Camera connection
        self.camera = None
        # Bounds for color masks
        self.lower_g = np.array([50, 102, 61])
        self.upper_g = np.array([80, 255, 125])
        self.lower_y = np.array([19, 54, 234])
        self.upper_y = np.array([69, 255, 255])
        self.lower_r = np.array([0, 180, 177])
        self.upper_r = np.array([179, 255, 225])

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
        yi = [np.matmul(np.linalg.inv(self.K), np.array([PMi[i][0], PMi[i][1], 1])) \
              for i in range(len(PMi))]
        I = np.eye(3)
        C = np.zeros((3*len(PMi), 13))
        for i in range(len(PMi)):
            kron = np.kron(I, np.array([PMi[i][0], PMi[i][1], 1]))
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

    def identifyTarget(self, image, color, display=False):
        if color == "green":
            lower = self.lower_g
            upper = self.upper_g
        elif color == "yellow":
            lower = self.lower_y
            upper = self.upper_y
        elif color == "red":
            lower = self.lower_r
            upper = self.upper_r
        else:
            raise ValueError
        allpoints = []
        approxpoints = []
        # Convert Image to HSV and apply mask(s) for colors
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        if display:
            cv2.imshow("Mask", mask)
        # Find contours
        cntrs, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Find corners on contours
        for i in cntrs:
            size = cv2.contourArea(i)
            rect = cv2.minAreaRect(i)
            if 1000<size<10000:
                gray = np.float32(mask)
                mask = np.zeros(gray.shape, dtype="uint8")
                cv2.fillPoly(mask, [i], (255, 255, 255))
                dst = cv2.cornerHarris(mask, 5, 3, 0.04)
                ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
                dst = np.uint8(dst)
                ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
                allpoints.extend([tuple(i) for i in corners])
                for eps in np.linspace(0.001, 0.05, 10):
                    # approximate the contour
                    peri = cv2.arcLength(i, True)
                    approx = cv2.approxPolyDP(i, eps * peri, True)
                approxpoints.extend([tuple(i[0]) for i in approx])
                if display is True:
                    for p in approxpoints:
                        cv2.circle(image, p, 5, (0, 255, 0), 2)
                    cv2.drawContours(image, [approx], -1, (255, 0, 0), 3)
                    image[dst > 0.1 * dst.max()] = [0, 0, 255]
                    cv2.imshow('image', image)
                    cv2.waitKey(0)
        return (allpoints, approxpoints)


if __name__ == '__main__':
    cnt = CameraController()
    im = cv2.imread("Blocks/allblocks.png")
    (points, rect) = cnt.identifyTarget(im, "red", True)
    print("Points")
    print(points)
    print("Approximated Points")
    print(rect)
    A = cnt.targetCameraPosition(points)
    print("Rotation Actual Points")
    print(A[0])
    print("Position Actual Points")
    print(A[1])
    B = cnt.targetCameraPosition(rect)
    print("Rotation Approx Points")
    print(B[0])
    print("Position Approx Points")
    print(B[1])
    print("Rotation Difference")
    print(np.subtract(A[0], B[0]))
    print("Position Difference")
    print(np.subtract(A[1], B[1]))



