import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

print("OpenCV version:", cv.__version__)
print("NumPy version:", np.__version__)
print("Opening image...")

img = cv.imread("media\Image__1.bmp")

if img is None:
    print("Error: Could not load image!")
    exit()

# Matplotlib verwacht RGB → dus converteren
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

def nothing(x):
    pass

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.namedWindow("Trackbars")

cv.createTrackbar("H_low", "Trackbars", 0, 179, nothing)
cv.createTrackbar("H_high", "Trackbars", 179, 179, nothing)
cv.createTrackbar("S_low", "Trackbars", 0, 255, nothing)
cv.createTrackbar("S_high", "Trackbars", 255, 255, nothing)
cv.createTrackbar("V_low", "Trackbars", 0, 255, nothing)
cv.createTrackbar("V_high", "Trackbars", 255, 255, nothing)

while True:
    H_low = cv.getTrackbarPos("H_low", "Trackbars")
    H_high = cv.getTrackbarPos("H_high", "Trackbars")
    S_low = cv.getTrackbarPos("S_low", "Trackbars")
    S_high = cv.getTrackbarPos("S_high", "Trackbars")
    V_low = cv.getTrackbarPos("V_low", "Trackbars")
    V_high = cv.getTrackbarPos("V_high", "Trackbars")

    lower = np.array([H_low, S_low, V_low])
    upper = np.array([H_high, S_high, V_high])

    mask = cv.inRange(hsv, lower, upper)
    result = cv.bitwise_and(img, img, mask=mask)

    cv.imshow("Mask", mask)
    cv.imshow("Result", result)

    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()


print("Done.")