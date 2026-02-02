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

edges = cv.Canny(img, threshold1=100, threshold2=50)

# Resultaten laten zien
plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Origineel")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edges)
plt.title("Edge Image")
plt.axis("off")

plt.show()