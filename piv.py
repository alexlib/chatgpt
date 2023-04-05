import cv2
import numpy as np

def perform_piv(img1, img2, window_size=16, stride=8):
    """
    Performs PIV on two consecutive images using optical flow method.
    :param image1: First input image.
    :param image2: Second input image.
    :param window_size: Size of the search window.
    :param stride: Stride of the search window.
    :return: u, v: velocity vectors.
    """
    # Convert images to grayscale
    # img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Define parameters for optical flow algorithm
    lk_params = dict(winSize=(window_size, window_size), 
                     maxLevel=4, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Initialize velocity vectors
    u, v = np.zeros_like(img1), np.zeros_like(img2)
    
    # Extract velocity vectors from optical flow
    for i in range(0, img1.shape[0] - window_size, stride):
        for j in range(0, img1.shape[1] - window_size, stride):
            # Define search window for optical flow
            x1, y1, x2, y2 = j, i, j + window_size, i + window_size
            # Extract optical flow vectors within search window
            flow_window = flow[y1:y2, x1:x2]
            # Calculate average velocity vector within search window
            u[i:i+window_size, j:j+window_size] = np.mean(flow_window[:,:,0])
            v[i:i+window_size, j:j+window_size] = np.mean(flow_window[:,:,1])
    
    return u, v
