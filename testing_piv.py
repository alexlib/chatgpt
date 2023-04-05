import numpy as np
import cv2
import matplotlib.pyplot as plt
from piv import perform_piv

# Generate ground truth vector field (vortex)
x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)
u = 5 * (1 - np.exp(-r**2 / 5**2)) * np.sin(theta)
v = -5 * (1 - np.exp(-r**2 / 5**2)) * np.cos(theta)
vector_field = np.dstack((u, v))

# Generate synthetic images by shifting vortex pattern
img1 = np.zeros((100, 100))
img2 = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        img1[i, j] = np.sin(u[i, j] + v[i, j])
        img2[i, j] = np.sin(u[i, j] + v[i, j] + 0.1)

# Apply PIV algorithm to images
u_piv, v_piv = perform_piv(img1, img2)

# Calculate error between computed and ground truth vector fields
error = np.sqrt((u_piv - u) ** 2 + (v_piv - v) ** 2)

# Plot results
fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
ax[0].imshow(img1, cmap='gray')
ax[0].quiver(u, v, color='red')
ax[0].set_title('Image 1')
ax[1].imshow(img2, cmap='gray')
ax[1].quiver(u_piv, v_piv, color='red')
ax[1].quiver(u, v, color='blue')
ax[1].set_title('Image 2')
plt.show()
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img1, cmap='gray')
ax.quiver(u_piv, v_piv, color='red')
ax.quiver(u, v, color='blue')
ax.set_title('PIV Results')
plt.show()

# Print mean error
print('Error: ', np.mean(error))
