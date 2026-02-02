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

for i, col in enumerate(("b", "g", "r")):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.title("Color Histogram")
plt.show()
