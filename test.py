import numpy as np
import cv2

# Generate synthetic image with 10x10 grid of random vectors
img_size = (256, 256)  # image size
grid_size = 10  # number of grid points in x and y directions
grid_spacing = img_size[0] // (grid_size + 1)  # grid point spacing

# Generate grid of random vectors
vectors = np.random.randn(grid_size, grid_size, 2)
x, y = np.meshgrid(np.arange(grid_spacing, img_size[0] - grid_spacing, grid_spacing),
                   np.arange(grid_spacing, img_size[1] - grid_spacing, grid_spacing), indexing='xy')
pts1 = np.stack([x, y], axis=2)
pts2 = pts1 + vectors

# Draw vectors on image
img1 = np.zeros(img_size, dtype=np.uint8)
img2 = np.zeros(img_size, dtype=np.uint8)
for i in range(grid_size):
    for j in range(grid_size):
        cv2.line(img1, tuple(pts1[i, j]), tuple(pts1[i, j] + vectors[i, j]), (255, 255, 255), 1)
        cv2.line(img2, tuple(pts2[i, j]), tuple(pts2[i, j] - vectors[i, j]), (255, 255, 255), 1)
