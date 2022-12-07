import Dofbot as bot
import CameraController as cm
import numpy as np
import time as time


class SystemController:
    def __init__(self):
        """
        Constructor for the controller class that controls the arm behavior
        """
        self.currentState = 0
        self.defaultAngles = np.array([0, 0, 0, 0, 0])
        self.approximatePayload = np.array([0, 0, 0, 0, 0])
        self.approximateDropOff = np.array([45, 125, 0, 0, 90])
        self.arm = bot.Dofbot()
        self.camera = cm.CameraController()
        # Path Planning Variables
        self.N = 100
        self.epsilon_p = 0.1
        self.kp = 0.055
        self.ki = 0.02
        self.kd = 0

    def run(self):
        # Run startup sequence
        self.startUp()
        while True:
            if self.currentState == 0:
                self.moveToObject()
            elif self.currentState == 1:
                self.pickupObject()
            elif self.currentState == 2:
                self.moveToDropoff()
            elif self.currentState == 3:
                self.dropOffObject()
            else:
                break

    def moveToObject(self):
        # Move to approximate payload location
        q = self.arm.readAllServoAngles()
        Rd, Pd = self.arm.getPositionFromAngles(self.approximatePayload)
        # TODO: Move along generated path
        # Change State
        self.currentState = 1

    def pickupObject(self):
        """
        Function for the pickup object state, that identifies the object, moves the arm to the object, and picks it up
        """
        # Take Picture of object to pickup
        res = False
        im = None
        for i in range(0, 5):
            res, im = self.camera.takePicture()
            if res:
                break
            elif i is 4:
                exit("Failed to take picture")
        # Identify Location of object
        Real, PMi = self.camera.identifyTarget(im, "green")
        RCM, PCM, w = self.camera.targetCameraPosition(PMi)
        q = self.arm.readAllServoAngles()
        R0T, P0T = self.arm.getPositionFromAngles(q)
        R0M, P0M = self.camera.targetBasePosition(RCM, PCM, R0T, P0T, q)
        # Move to object and pick it up
        self.arm.openClaw()
        # TODO: Move along generated path
        self.arm.closeClaw()
        # Change State
        self.currentState = 2

    def moveToDropoff(self):
        # Move to drop off location
        q = self.arm.readAllServoAngles()
        Rd, Pd = self.arm.getPositionFromAngles(self.approximateDropOff)
        # TODO: Move along generated path
        # Change State
        self.currentState = 3

    def dropOffObject(self):
        # Drop off object in box
        self.arm.openClaw()
        time.sleep(0.1)
        self.arm.closeClaw()
        # Extra: Determine box location (CV may be necessary)
        # Extra: Move arm to appropriate location
        # Extra: Put object down
        # Extra: return to drop off location
        # Change State
        self.currentState = 0

    def startUp(self) -> None:
        # Connect to Dofbot arm
        self.arm.connect()
        # Connect to Camera
        self.camera.connectCamera()
        # Move arm to default Position
        self.arm.setAllServoAngles(self.defaultAngles, 0.0)
        return None
