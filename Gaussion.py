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

# Verschillende Gaussian blur opties
gauss_3 = cv.GaussianBlur(img, (3, 3), 0)
gauss_5 = cv.GaussianBlur(img, (5, 5), 0)
gauss_7 = cv.GaussianBlur(img, (7, 7), 0)

# Converteren naar RGB voor matplotlib
gauss_3 = cv.cvtColor(gauss_3, cv.COLOR_BGR2RGB)
gauss_5 = cv.cvtColor(gauss_5, cv.COLOR_BGR2RGB)
gauss_7 = cv.cvtColor(gauss_7, cv.COLOR_BGR2RGB)

# Resultaten laten zien
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Origineel")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gauss_3)
plt.title("Gaussian Blur 3x3")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(gauss_5)
plt.title("Gaussian Blur 5x5")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(gauss_7)
plt.title("Gaussian Blur 7x7")
plt.axis("off")

plt.tight_layout()
plt.show()
