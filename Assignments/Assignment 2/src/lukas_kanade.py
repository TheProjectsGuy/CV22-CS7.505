# Lukas-Kanade (LK) algorithm
"""
    Implements the Lukas-Kanade algorithm as described in [1] (
    implemented towards optical flow).
    Official OpenCV tutorial for Optical Flow at [2]
    The following are developed through OpenCV
    - cartToPolar [3]: Cartesian to Polar coordinates
    - Sobel [4][5]: To get the image gradients (X and Y)
    - GaussianBlur [6][7]: Applied to the gradients (window)

    The following functions can be looked at, to replicate results 
    using OpenCV
    - calcOpticalFlowFarneback [8][10]: Dense Optical Flow using 
        OpenCV
    - calcOpticalFlowPyrLK [9][11]: Optical Flow using Pyramid Lukas-
        Kanade algorithm

    [1]: Lucas, Bruce D., and Takeo Kanade. "An iterative image registration technique with an application to stereo vision." 1981.
    [2]: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
    [3]: https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gac5f92f48ec32cacf5275969c33ee837d
    [4]: https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
    [5]: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
    [6]: https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    [7]: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    [8]: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
    [9]: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
    [10]: https://github.com/opencv/opencv/blob/4.x/modules/video/src/optflowgf.cpp#L1194
    [11]: https://github.com/opencv/opencv/blob/4.x/modules/video/src/lkpyramid.cpp#L1411
"""

# %% Import everything
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# Utilities
import os
from utils import imrescale_uint8, imrescale_unit

# %%
# Lukas-Kanade algorithm in a grid
def lukas_kanade_grid(img_f, img_g, win_h = 25, win_w = 25, 
        ev_thresh = 1.0):
    """
        Lukas-Kanade algorithm on a grid (much faster). Returns the
        optical flow as an array, along with grid positions. The grid
        is decided by the window (spacing).

        Parameters:
        - img_f: np.ndarray         shape: H, W
            Image 1 (initial) in grayscale
        - img_g: np.ndarray         shape: H, W
            Image 2 (final) in grayscale
        - win_h: int
            The height of window
        - win_w: int
            The width of window
        - ev_thresh: float
            The threshold for eigen-values. The individual values are
            also returned.
        
        Returns:
        - positions: np.ndarray         shape: N, 2
            The [height, width] position of the returned optical flow 
            vectors.
        - optical_flow: np.ndarray      shape: N, 2
            The [X, Y] values for each optical flow vector. There are
            N optical flow vectors.
        - thresh_val: np.ndarray        shape: N, 
            Threshold values for each optical flow vector.
    """
    # Rescale to 0 to 1
    img_f = imrescale_unit(img_f)
    img_g = imrescale_unit(img_g)
    # G - F differential image
    dimg = img_g - img_f
    # Sobel operation (for dx and dy of 'f')
    Ifx = cv.Sobel(img_f, -1, 1 ,0)
    Ify = cv.Sobel(img_f, -1, 0 ,1)
    Ifx2 = (Ifx * Ifx)
    Ify2 = (Ify * Ify)
    Ifxy = (Ifx * Ify)
    # Short-hand matrices
    A = Ifx2
    B = Ify2
    C = Ifxy
    D = dimg * Ifx
    E = dimg * Ify
    # Optical flow calculations
    positions = []
    optical_flow = []
    thresh_val = []
    nf = win_h * win_w  # Normalizing
    imH, imW = dimg.shape
    for i in range(0, imH // win_h, 1):
        for j in range(0, imW // win_w, 1):
            # Ranges
            r_st, r_en = i*win_h, (i+1)*win_h
            c_st, c_en = j*win_w, (j+1)*win_w
            # Numerator
            nx = np.sum(D[r_st:r_en, c_st:c_en])
            ny = np.sum(E[r_st:r_en, c_st:c_en])
            n = np.array([[nx, ny]])
            # Denominator
            dxx = np.sum(A[r_st:r_en, c_st:c_en])
            dyy = np.sum(B[r_st:r_en, c_st:c_en])
            dxy = np.sum(C[r_st:r_en, c_st:c_en])
            mat_A = np.array([[dxx, dxy], [dxy, dyy]])
            # Inverse of denominator
            evals, _ = np.linalg.eig(mat_A.T @ mat_A)
            min_ev = np.min(evals)
            tc = min_ev/nf  # Compare this
            # Register optical flow
            h_val = [0, 0]
            thresh_val.append(tc)   # Threshold value
            if tc > ev_thresh:
                h_val = (n @ np.linalg.inv(mat_A)).flatten().tolist()
            positions.append([i*win_h + win_h//2, j*win_w + win_w//2])
            optical_flow.append(h_val)
    positions_np = np.array(positions)
    optical_flow_np = np.array(optical_flow)
    thresh_val_np = np.array(thresh_val)
    # Invert (-1) Y, then invert the whole vector (== invert X)
    optical_flow_np[:, 0] *= -1
    # Optical flow values
    return positions_np, optical_flow_np, thresh_val_np


# %%
# Lukas-Kanade algorithm through a dense grid setting
def lukas_kanade_dense(img_f, img_g, win_h = 25, win_w = 25, 
        ev_thresh = 1.0):
    """
        Lukas-Kanade algorithm for a dense correspondence (every 
        pixel). Returns the optical flow and thresholds as images.

        Parameters:
        - img_f: np.ndarray         shape: H, W
            Image 1 (initial) in grayscale
        - img_g: np.ndarray         shape: H, W
            Image 2 (final) in grayscale
        - win_h: int
            The height of window
        - win_w: int
            The width of window
        - ev_thresh: float
            The threshold for eigen-values. The individual values are
            also returned.
        
        Returns:
        - opt_flow: np.ndarray      shape: H, W, 2
            The optical flow. Indices [..., 0] is X and [..., 1] is Y
        - th_img: np.ndarray        shape: H, W
            The threshold image (storing the threshold values)
    """
    # Rescale to 0 to 1
    img_f = imrescale_unit(img_f)
    img_g = imrescale_unit(img_g)
    # G - F differential image
    dimg = img_g - img_f
    # Sobel operation (for dx and dy of 'f')
    Ifx = cv.Sobel(img_f, -1, 1 ,0)
    Ify = cv.Sobel(img_f, -1, 0 ,1)
    # Higher order gradients
    Ifx2 = (Ifx * Ifx)
    Ify2 = (Ify * Ify)
    Ifxy = (Ifx * Ify)
    # Short-hand matrices
    A = Ifx2
    B = Ify2
    C = Ifxy
    D = dimg * Ifx
    E = dimg * Ify
    # Optical flow calculations
    nf = win_h * win_w  # Normalizing
    imH, imW = dimg.shape
    opt_flow = np.zeros((imH, imW, 2), float)   # Pixel: X, Y
    th_img = np.zeros((imH, imW), float)    # Threshold image
    for i in range(0, imH - win_h, 1):
        for j in range(0, imW - win_w, 1):
            # Ranges
            r_st, r_en = i, i+win_h
            c_st, c_en = j, j+win_w
            # Numerator
            nx = np.sum(D[r_st:r_en, c_st:c_en])
            ny = np.sum(E[r_st:r_en, c_st:c_en])
            n = np.array([[nx, ny]])
            # Denominator
            dxx = np.sum(A[r_st:r_en, c_st:c_en])
            dyy = np.sum(B[r_st:r_en, c_st:c_en])
            dxy = np.sum(C[r_st:r_en, c_st:c_en])
            mat_A = np.array([[dxx, dxy], [dxy, dyy]])
            # Inverse of denominator
            evals, _ = np.linalg.eig(mat_A.T @ mat_A)
            min_ev = np.min(evals)
            tc = min_ev/nf  # Compare this
            # Register optical flow
            h_val = [0, 0]
            if tc > ev_thresh:
                h_val = (n @ np.linalg.inv(mat_A)).flatten().tolist()
            # Log
            opt_flow[i+win_h//2, j+win_w//2] = h_val  # Optical Flow
            th_img[i+win_h//2, j+win_w//2] = tc       # Threshold
    # Invert (-1) Y, then invert the whole vector (== invert X)
    opt_flow[..., 0] *= -1
    # Results
    return opt_flow, th_img


# %% Optical Flow to HSV & RGB image
def opt_flow_to_imgs(opt_flow, mag_thresh = 0.05):
    """
        Converts an optical flow to HSV and BGR images that can be
        visualized. Hue gets mapped to angle, Saturation is 255, and
        Value is the magnitude. Angle and magnitude is the polar form
        of the optical flow.

        Parameters:
        - opt_flow: np.ndarray      shape: H, W, 2
            The optical flow. Indices [..., 0] is X and [..., 1] is Y
        - mag_thresh: float
            The magnitude for thresholding (polar radius). Values
            under this are made 0.

        Returns:
        - hsv_img: np.ndarray       shape: H, W, 3
            HSV image. H: 0 to 180, S: 0 to 255, V: 0 to 255 (uint8)
        - bgr_img: np.ndarray       shape: H, W, 3
            BGR images (converted from HSV using `cvtColor`)
    """
    imH, imW, _ = opt_flow.shape
    hsv_img = np.zeros((imH, imW, 3), np.uint8) # HSV image
    # Convert X, Y (cartesian) to r, theta (polar)
    mag, ang = cv.cartToPolar(opt_flow[..., 0], opt_flow[..., 1])
    mag[mag < mag_thresh] = 0   # Prune magnitude (reduce noise)
    mag_ui8 = imrescale_uint8(mag)  # Magnitude 0 to 255 uint8
    ang_180 = ang * (180/(2*np.pi)) # Angle 0 to 180 float
    # Fill data in HSV image
    hsv_img[..., 0] = ang_180.astype(np.uint8)  # Hue - Angle
    hsv_img[..., 1] = 255   # Full saturation
    hsv_img[..., 2] = mag_ui8   # Magnitude is brightness
    # BGR image
    bgr_img = cv.cvtColor(hsv_img.copy(), cv.COLOR_HSV2BGR)
    # Return images
    return hsv_img, bgr_img


# %% OpenCV Optical Flow - Dense
def optical_flow_cvdense(img_f, img_g, prior_flow = None, 
        pyr_scale = 0.5, levels = 3, winsize = 15, iters = 3, 
        poly_n = 5, poly_sigma = 1.2, flags = 0):
    """
        Farneback Optical Flow algorithm in OpenCV for dense optical
        flow (every pixel). Returns the optical flow. Algorithm 
        options can be found at [1]

        Parameters:
        - img_f: np.ndarray         shape: H, W
            Image 1 (initial) in grayscale
        - img_g: np.ndarray         shape: H, W
            Image 2 (final) in grayscale
        - prior_flow: np.ndarray or None    shape: H, W, 2
            The prior optical flow for the algorithm (could be None)
        - pyr_scale: float
            Pyramid scale for the algorithm
        - levels: float
            Number of pyramid levels
        - winsize: int
            Window size for averaging
        - iters: int
            Number of iterations for algorithm (at each pyramid level)
        - poly_n: int
            Pixel neighborhood for polynomial expansion
        - poly_sigma: float
            Standard deviation of Gaussian for polynomial derivatives
        - flags: int
            Flags for the algorithm
        
        Returns:
        - flow_cv: np.ndarray
            The optical flow. Indices [..., 0] is X and [..., 1] is Y

        [1]: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
    """
    prior = None
    if prior_flow is not None:
        prior = np.array(prior_flow, np.float32)    # Prior flow
    # Optical flow through OpenCV
    flow_cv = cv.calcOpticalFlowFarneback(img_f, img_g, prior, 
        pyr_scale, 3, 15, 3, 5, 1.2, 0).astype(float)
    return flow_cv


# %% Main function
if __name__ == "__main__":
    img1 = cv.imread(
        "./../data/all-frames-colour/RubberWhale/frame07.png")
    img2 = cv.imread(
        "./../data/all-frames-colour/RubberWhale/frame08.png")
    # img1 = cv.imread("./../data/test/img1.jpg")
    # img2 = cv.imread("./../data/test/img2.jpg")
    # Show image
    plt.figure(figsize=(10, 5))
    plt.suptitle("Images")
    plt.subplot(1,2,1)
    plt.title("Image 1")
    plt.imshow(img1[..., ::-1])
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Image 2")
    plt.imshow(img2[..., ::-1])
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # Optical Flow: Grid
    positions_np, optical_flow_np, tvs = lukas_kanade_grid(
        cv.cvtColor(img1.copy(), cv.COLOR_BGR2GRAY), 
        cv.cvtColor(img2.copy(), cv.COLOR_BGR2GRAY), 
        win_h = 25, win_w = 25, ev_thresh = 0.05)
    # Show image
    plt.title("Grid Optical Flow")
    plt.imshow(img1[..., ::-1])
    for pos, h, tv in zip(positions_np, optical_flow_np, tvs):
        # print(f"{tv}")
        if np.allclose(h, np.zeros_like(h)):
            plt.plot(pos[1], pos[0], 'r.')
        else:
            plt.quiver(pos[1], pos[0], h[0], h[1], color='red', 
                width=5e-3, scale=3)
    plt.tight_layout()
    plt.show()
    # Optical Flow: Dense
    opt_flow, th_img = lukas_kanade_dense(
        cv.cvtColor(img1.copy(), cv.COLOR_BGR2GRAY), 
        cv.cvtColor(img2.copy(), cv.COLOR_BGR2GRAY), 15)
    # Show images
    plt.figure(figsize=(18, 6))
    plt.suptitle("Dense Optical Flow")
    plt.subplot(1,2,1)
    plt.title("X")
    plt.imshow(opt_flow[..., 0], 'gray')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.title("Y")
    plt.imshow(opt_flow[..., 1], 'gray')
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    # Get HSV and BGR
    hsv_img, bgr_img = opt_flow_to_imgs(opt_flow)
    plt.title("Optical Flow - HSV")
    plt.imshow(bgr_img[..., ::-1])
    plt.show()
    # Optical Flow: Dense: OpenCV
    opt_flow_cv = optical_flow_cvdense(
        cv.cvtColor(img1.copy(), cv.COLOR_BGR2GRAY),
        cv.cvtColor(img2.copy(), cv.COLOR_BGR2GRAY))
    hsv_cv, bgr_cv = opt_flow_to_imgs(opt_flow_cv, 0.0)
    plt.title("Optical Flow - HSV - OpenCV")
    plt.imshow(bgr_cv[..., ::-1])
    plt.show()


# %% Experimental section
