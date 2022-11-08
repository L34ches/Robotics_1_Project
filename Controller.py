import Dofbot as bot
import numpy as np


class SystenController:
    def __init__(self):
        """
        Constructor for the controller class that controls the arm behavior
        """
        self.currentState = 0
        self.defaultAngles = np.array([0, 0, 0, 0, 0])
        self.approximatePayload = np.array([0, 0, 0, 0, 0])
        self.approximateDropOff = np.array([0, 0, 0, 0, 0])
        self.arm = bot.Dofbot()

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
        # Pickup Object
        # TODO: Take Picture of object to pickup
        # TODO: Identify Location of object
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
        # Move arm to default Position
        self.arm.setAllServoAngles(self.defaultAngles, 0.0)
        return None
