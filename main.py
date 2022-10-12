import numpy as np
import NoConnectionDofbot as db


if __name__ == "__main__":
    bot = db.NoConnectDofbot()

    # Test Forward Kinematic Implementation
    q = np.array([90, 90, 90, 90, 90])
    Result = bot.getPositionFromAngles(q)
    print("Case 1: q=[90, 90, 90, 90, 90]")
    print("Rotation:\n", Result[0])
    print("Position:\n", Result[1])
    print("Case 2: q=[0, 45, 135, 45, 135]")
    q = np.array([0, 45, 135, 45, 135])
    Result = bot.getPositionFromAngles(q)
    print("Rotation:\n", Result[0])
    print("Position:\n", Result[1])

