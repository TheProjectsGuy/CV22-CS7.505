# Implements Harris Corner Detector
"""
    Implements the Harris Corner Detector as described in [1]. Aims to
    answer part '(i)' in section 2.1: "Keypoint Selection: Selecting
    Pixels to Track"
    The following are implemented through OpenCV
    - Sobel [4][5]: To get the image gradients (X and Y).
    - GaussianBlur [2][3]: Applied to the gradients (window).

    Official OpenCV tutorial at [6]

    [1]: Harris, Chris, and Mike Stephens. "A combined corner and edge detector." Alvey vision conference. Vol. 15. No. 50. 1988.
    [2]: https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    [3]: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    [4]: https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
    [5]: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
    [6]: https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
"""

# %% Import everything
import numpy as np
import cv2 as cv
# from scipy.signal import convolve2d
from matplotlib import pyplot as plt
# Utilities
# from utils import imrescale_uint8


# %% Function for main harris corner (implemented from ground)
def harris_corner_detector(img, k = 0.04, gk_size = 5,
        fil = True, min_detect = 0.02, pix_thr = 5):
    """
        Runs the harris corner detector. Returns the points (corners).
        Implemented the algorithm from ground (as in the paper).

        Parameters:
        - img: np.ndarray       shape: H, W
            The greyscale image (as numpy array)
        - k: float
            The 'k' weight for trace**2 in the corner response
        - gk_size: int
            The size of the gaussian kernel for blurring the gradients
        - fil: bool
            If 'True' then filter the points, else (if 'False') do not
            filter the points (many duplicates may exist)
        - min_detect: float
            The minimum detection threshold for the corner response
        - pix_thr: int
            The pixel threshold for filtering. If there are many 
            corner points in this threshold, they're reduced to one.
            This could still lead to repeated corners in some cases.

        Returns:
        - pts: np.ndarray       shape: N, 2
            The corner points [x, y] in an array
    """
    # Sobel operation (for dx, dy)
    Ix = cv.Sobel(img, cv.CV_16S, 1, 0)
    Iy = cv.Sobel(img, cv.CV_16S, 0, 1)
    # Three elements for shifting error
    Ix2 = (Ix.astype(float) * Ix.astype(float))
    Iy2 = (Iy.astype(float) * Iy.astype(float))
    IxIy = (Ix.astype(float) * Iy.astype(float))
    # Gaussian Blur to the gradients (for window)
    kernel = (gk_size, gk_size)
    A = cv.GaussianBlur(Ix2, kernel, 0)
    B = cv.GaussianBlur(Iy2, kernel, 0)
    C = cv.GaussianBlur(IxIy, kernel, 0)
    # Corner response
    M_det = A*B - (C**2)
    M_tr = A + B
    R = M_det - k * (M_tr)**2
    # Get corners only
    R_corners = R.copy()
    R_corners[R_corners < 0] = 0    # Only positive will remain
    # Normalize them (0 to 1)
    R_corners = (R_corners - R_corners.min())/R_corners.ptp()
    # All corner points
    all_c = np.where(R_corners >= min_detect)   # [y, x]
    if fil: # Filtering
        fpts = [    # Filtered points (clusters)
            [all_c[0][0], all_c[1][0]]  # Y, X
        ]
        # Euclidean distance
        eu_dist = lambda p1, p2: \
            (((p2[1]-p1[1])**2) + ((p2[0]-p1[0])**2))**0.5
        # Filtering process
        for py, px in zip(*all_c):
            dpx = np.array(fpts) - \
                np.array([[py, px]])     # len(fpts), 2
            dist_vals = np.linalg.norm(dpx, 
                axis=1) # shape: len(fpts),
            cl_ind = np.argmin(dist_vals)   # Closest index in 'fpts'
            if eu_dist(fpts[cl_ind], [py, px]) > pix_thr:
                # New point
                fpts.append([py, px])
            else:
                ay, ax = fpts[cl_ind]   # Already in filtered points
                if R_corners[py, px] > R_corners[ay, ax]:
                    # Higher intensity, better alternative
                    fpts[cl_ind] = [py, px]
        # Result
        res = np.array(fpts)[:,[1,0]]
    else:   # No filtering
        res = np.stack((all_c[1], all_c[0])).T  # Result
    return res

# %% Function for harris corner (using OpenCV)
def harris_corner_opencv(img, bls = 3, ks = 3, fil = True, 
        min_detect = 0.15, pix_thr = 5):
    """
        Harris corner implementation through OpenCV.

        Parameters:
        - img: np.ndarray       shape: H, W
            The greyscale image (as numpy array)
        - bls: int
            The block size (for eigenvalue and eigenvector in OpenCV)
        - ks: int
            The kernel size (used by OpenCV)
        - fil: bool
            If 'True' then filter the points, else (if 'False') do not
            filter the points (many duplicates may exist)
        - min_detect: float
            The minimum detection threshold for the corner response
        - pix_thr: int
            The pixel threshold for filtering. If there are many 
            corner points in this threshold, they're reduced to one.
            This could still lead to repeated corners in some cases.
        
        Returns:
        - pts: np.ndarray       shape: N, 2
            The corner points [x, y] in an array
    """
    # Get the eigenvalues and eigenvectors
    dst = cv.cornerEigenValsAndVecs(img, bls, ks)
    eig_vals = dst[..., 0:2]    # Eigenvalues
    # Harris operator
    f = np.prod(eig_vals, axis=2)/(np.sum(eig_vals, axis=2) + 1e-5)
    R_corners = (f - f.min())/f.ptp()   # Normalized
    all_c = np.where(R_corners >= min_detect)   # [y, x]
    if fil: # Filtering
        fpts = [    # Filtered points (clusters)
            [all_c[0][0], all_c[1][0]]  # Y, X
        ]
        # Euclidean distance
        eu_dist = lambda p1, p2: \
            (((p2[1]-p1[1])**2) + ((p2[0]-p1[0])**2))**0.5
        # Filtering process
        for py, px in zip(*all_c):
            dpx = np.array(fpts) - \
                np.array([[py, px]])     # len(fpts), 2
            dist_vals = np.linalg.norm(dpx, 
                axis=1) # shape: len(fpts),
            cl_ind = np.argmin(dist_vals)   # Closest index in 'fpts'
            if eu_dist(fpts[cl_ind], [py, px]) > pix_thr:
                # New point
                fpts.append([py, px])
            else:
                ay, ax = fpts[cl_ind]   # Already in filtered points
                if R_corners[py, px] > R_corners[ay, ax]:
                    # Higher intensity, better alternative
                    fpts[cl_ind] = [py, px]
        # Result
        res = np.array(fpts)[:,[1,0]]
    else:   # No filtering
        res = np.stack((all_c[1], all_c[0])).T  # Result
    return res

# %% Main entrypoint
if __name__ == "__main__":
    # Image
    img_file = "./../data/test/blox.jpg"
    img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
    # OpenCV implementation
    res = harris_corner_opencv(img)
    # Local implementation
    res_local = harris_corner_detector(img)
    # Image 1
    img_c = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
    for px, py in res:
        cv.circle(img_c, (px, py), 3, (255, 0, 0), cv.FILLED)
    # Image 2
    img_cl = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
    for px, py in res_local:
        cv.circle(img_cl, (px, py), 3, (255, 0, 0), cv.FILLED)
    # Plots
    plt.subplot(1,2,1)
    plt.title("OpenCV")
    plt.imshow(img_c)
    plt.subplot(1,2,2)
    plt.title("Custom")
    plt.imshow(img_cl)

# %% Experimental section

# %%
