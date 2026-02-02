import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("media\Image__1.bmp")

if img is None:
    print("Error loading image")
    exit()

# --- Enhancement ---
gauss = cv.GaussianBlur(img, (5, 5), 0)
gauss7 = cv.GaussianBlur(img, (7, 7), 0)

# Sharpening filter
kernel_sharp = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
], dtype=np.float32)
sharp = cv.filter2D(img, -1, kernel_sharp)

# Matplotlib verwacht RGB → dus converteren
img_rgb = cv.cvtColor(gauss, cv.COLOR_BGR2RGB)

def nothing(x):
    pass

hsv = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV)
cv.namedWindow("Trackbars")

cv.createTrackbar("H_low", "Trackbars", 0, 179, nothing)
cv.createTrackbar("H_high", "Trackbars", 94, 179, nothing)
cv.createTrackbar("S_low", "Trackbars", 29, 255, nothing)
cv.createTrackbar("S_high", "Trackbars", 255, 255, nothing)
cv.createTrackbar("V_low", "Trackbars", 97, 255, nothing)
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
    result = cv.bitwise_and(gauss, gauss, mask=mask)

    cv.imshow("Mask", mask)
    cv.imshow("Result", result)

    if cv.waitKey(1) & 0xFF == 27: # Press escape to
        break

cv.destroyAllWindows()


# Canny edges
edges = cv.Canny(result, 100, 50)

print("Done.")

# Show results
plt.figure(figsize=(12,8))
plt.subplot(221), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title("Origineel")
plt.subplot(222), plt.imshow(cv.cvtColor(gauss, cv.COLOR_BGR2RGB)), plt.title("Gaussian 5x5")
plt.subplot(223), plt.imshow(cv.cvtColor(sharp, cv.COLOR_BGR2RGB)), plt.title("Sharpened")
plt.subplot(224), plt.imshow(edges, cmap='gray'), plt.title("Canny Edges")
plt.show()
