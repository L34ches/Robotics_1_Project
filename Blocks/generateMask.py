import cv2
import numpy as np

def empty(*aargs):
    pass


def generateMask():
    # Open image and create HSV copy
    im = cv2.imread("allblocks.png")
    cv2.imshow("A", im)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # Create trackbars to manipulate image
    cv2.namedWindow("Trackbar")
    cv2.resizeWindow("Trackbar", 600, 300)
    cv2.createTrackbar("hue_min", "Trackbar", 0, 179, empty)
    cv2.createTrackbar("hue_max", "Trackbar", 179, 179, empty)
    cv2.createTrackbar("sat_min", "Trackbar", 0, 255, empty)
    cv2.createTrackbar("sat_max", "Trackbar", 255, 255, empty)
    cv2.createTrackbar("val_min", "Trackbar", 0, 255, empty)
    cv2.createTrackbar("val_max", "Trackbar", 255, 255, empty)

    # Run loop to change image dynamically
    while True:
        # Get mask bounds from trackbars
        hue_min = cv2.getTrackbarPos("hue_min", "Trackbar")
        hue_max = cv2.getTrackbarPos("hue_max", "Trackbar")
        sat_min = cv2.getTrackbarPos("sat_min", "Trackbar")
        sat_max = cv2.getTrackbarPos("sat_max", "Trackbar")
        val_min = cv2.getTrackbarPos("val_min", "Trackbar")
        val_max = cv2.getTrackbarPos("val_max", "Trackbar")

        # Apply mask to image
        lower = np.array([hue_min, sat_min, val_min])
        upper = np.array([hue_max, sat_max, val_max])
        mask = cv2.inRange(hsv, lower, upper)
        cv2.imshow("HSV", mask)
        cv2.waitKey(1)


if __name__ == '__main__':
    generateMask()
