import Dofbot as bot
import CameraController as cm
import numpy as np
import quadprog as qp


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

    def run(self):
        # Run startup sequence
        self.startUp()
        while True:
            match self.currentState:
                case 0:
                    self.moveToObject()
                    continue
                case 1:
                    self.pickupObject()
                    continue
                case 2:
                    self.moveToDropoff()
                    continue
                case 3:
                    self.dropOffObject()
                    continue

    def moveToObject(self):
        # Move to approximate payload location
        self.arm.setAllServoAnglesVerified(self.approximatePayload, 0.0)
        # TODO: Change movement to path planned movement
        # Change State
        self.currentState = 1

    def pickupObject(self):
        """
        Function for the pickup object state, that identifies the object, moves the arm to the object, and picks it up
        """
        # Take Picture of object to pickup
        res, im = self.camera.takePicture()
        # Identify Location of object
        PMi = self.camera.identifyTarget(im, "green")
        RCM, PCM, w = self.camera.targetCameraPosition(PMi)
        q = self.arm.readAllServoAngles()
        R0T, P0T = self.arm.getPositionFromAngles(q)
        R0M, P0M = self.camera.targetBasePosition(RCM, PCM, R0T, P0T)
        # TODO: Plan Path to object
        # TODO: Move to object
        # TODO: Pickup Object
        # Change State
        self.currentState = 2

    def moveToDropoff(self):
        # Move to drop off location
        self.arm.setAllServoAnglesVerified(self.approximateDropOff, 0.0)
        # TODO: Change movement to path planned movement
        # Change State
        self.currentState = 3

    def dropOffObject(self):
        # Drop off object in box
        # TODO: Determine box location (CV may be necessary)
        # TODO: Move arm to appropriate location
        # TODO: Put object down
        # TODO: return to drop off location
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

    def planPath(self, Rc, Pc, Rd, Pd, *args):
        """
        Determine path from one position to another

        :param Rc: Current Rotation of arm
        :param Pc: Current Position of arm
        :param Rd: Desired Rotation of arm
        :param Pc: Desired Position of arm
        """
        # TODO: Implement path planning algorithm
