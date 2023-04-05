import numpy as np
import cv2


def create_images(img_size=(256, 256), n_particles=1000, particle_size=3, max_brightness=255):

    # Generate synthetic image with random bright dots
    img_size = (256, 256)  # image size
    n_particles = 1000  # number of particles
    particle_size = 3  # size of particles in pixels
    max_brightness = 255  # maximum brightness of particles

    # Create blank images
    img1 = np.zeros(img_size + (3,), dtype=np.uint8)
    img2 = np.zeros(img_size + (3,), dtype=np.uint8)

    # Randomly generate particle positions and brightnesses
    pts1 = np.random.uniform(
        particle_size // 2, img_size[0] - particle_size // 2, size=(n_particles, 2))
    brightnesses = np.random.randint(0, max_brightness + 1, size=n_particles)

    # Create ground truth vector field
    vector_field = np.random.randn(*img_size, 2)

    # Shift particle positions according to vector field
    pts2 = pts1 + vector_field[pts1[:,
                                    1].astype(int), pts1[:, 0].astype(int), :]

    # Draw particles on images
    for i, pt in enumerate(pts1):
        cv2.circle(img1, tuple(pt.astype(int)),
                   particle_size // 2, (255, 255, 255), -1)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    for i, pt in enumerate(pts2):
        cv2.circle(img2, tuple(pt.astype(int)),
                   particle_size // 2, (255, 255, 255), -1)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    return (img1, img2)
