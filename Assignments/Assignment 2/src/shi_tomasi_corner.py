# Implements Shi-Tomasi Corner Detector
"""
    Implements the Shi-Tomasi Corner Detector as described in [1]. 
    Aims to answer part '(ii)' in section 2.1: "Keypoint Selection: 
    Selecting Pixels to Track"
    The following are implemented through OpenCV
    - Sobel [5][6]: To get the image gradients (X and Y)
    - GaussianBlur [3][4]: Applied to the gradients (window)

    Official OpenCV tutorial at [2] (front-to-end implemented).

    [1]: Shi, Jianbo. "Good features to track." 1994 Proceedings of IEEE conference on computer vision and pattern recognition. IEEE, 1994.
    [2]: https://docs.opencv.org/4.x/d8/dd8/tutorial_good_features_to_track.html
    [3]: https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    [4]: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    [5]: https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
    [6]: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
"""

# %% Import everything
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# %% Function for main Shi-Tomasi Corner Detector (from ground)
def shi_tomasi_corner_detector(img, gk_size = 5, ev_thresh = 0.1, 
    fil = True, pix_thr = 5):
    """
        Runs the Shi-Tomasi Corner Detector corner detector. Returns 
        the points (corners). Implemented the algorithm from ground 
        (as in the paper).

        Parameters:
        - img: np.ndarray       shape: H, W
            The greyscale image (as numpy array)
        - gk_size: int
            Size of the gaussian kernel for blurring the gradients
        - ev_thresh: float
            The eigenvalue threshold (for the minima)
        - fil: bool
            If 'True' then filter the points, else (if 'False') do not
            filter the points (many duplicates may exist)
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
    # Generalized matrix formation (4x4 matrix - Z in paper)
    gen_mat = np.zeros([*A.shape, 2, 2])
    h, w, _, _ = gen_mat.shape
    for i in range(h):
        for j in range(w):
            gen_mat[i, j, 0, 0] = A[i, j]
            gen_mat[i, j, 0, 1] = C[i, j]
            gen_mat[i, j, 1, 0] = C[i, j]
            gen_mat[i, j, 1, 1] = B[i, j]
    # Get eigenvalues
    eig_vals, v = np.linalg.eig(gen_mat)
    min_ev = np.min(eig_vals, axis=2)   # Minimum of eigenvalues
    # Scale 0 to 1
    R_corners = (min_ev - min_ev.min())/min_ev.ptp()
    # Points passing thresh.
    all_c = np.where(R_corners > ev_thresh)
    fpts = [    # Filtered points (clusters)
        [all_c[0][0], all_c[1][0]]  # Y, X
    ]
    # Euclidean distance
    eu_dist = lambda p1, p2: \
        (((p2[1]-p1[1])**2) + ((p2[0]-p1[0])**2))**0.5
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


# %% Function for Shi-Tomasi corner (using OpenCV)
def shi_tomasi_corner_opencv(img, ev_thresh = 0.005, bls = 3, ks = 3,
        fil = True, pix_thr = 5):
    """
        Shi-Tomasi corner detector through OpenCV. Only eigen-value
        computation is offloaded to OpenCV. You can also use the
        function goodFeaturesToTrack [1]

        Parameters:
        - img: np.ndarray       shape: H, W
            The greyscale image (as numpy array)
        - ev_thresh: float
            The eigenvalue threshold (for the minima)
        - bls: int
            The block size (for eigenvalue and eigenvector in OpenCV)
        - ks: int
            The kernel size (used by OpenCV)
        - fil: bool
            If 'True' then filter the points, else (if 'False') do not
            filter the points (many duplicates may exist)
        - pix_thr: int
            The pixel threshold for filtering. If there are many 
            corner points in this threshold, they're reduced to one.
            This could still lead to repeated corners in some cases.
        
        Returns:
        - pts: np.ndarray       shape: N, 2
            The corner points [x, y] in an array
        
        [1]: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
    """
    # Get eigenvalues and eigenvectors
    dst = cv.cornerEigenValsAndVecs(img, bls, ks)
    eig_vals = dst[..., 0:2]    # Eigenvalues
    min_ev = np.min(eig_vals, axis=2)   # Minimum of eigenvalues
    all_c = np.where(min_ev > ev_thresh)    # Points passing thresh.
    fpts = [    # Filtered points (clusters)
        [all_c[0][0], all_c[1][0]]  # Y, X
    ]
    # Euclidean distance
    eu_dist = lambda p1, p2: \
        (((p2[1]-p1[1])**2) + ((p2[0]-p1[0])**2))**0.5
    R_corners = min_ev  # Response for corners
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

# %% Experimental section

# %%
if __name__ == "__main__":
    # Image
    img_file = "./../data/test/blox.jpg"
    img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
    # Using OpenCV
    res = shi_tomasi_corner_opencv(img)
    img_c = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
    for px, py in res:
        cv.circle(img_c, (px, py), 3, (255, 0, 0), cv.FILLED)
    img_cv = img_c.copy()
    # From ground-up
    res = shi_tomasi_corner_detector(img)
    img_c = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
    for px, py in res:
        cv.circle(img_c, (px, py), 3, (255, 0, 0), cv.FILLED)
    # Show everything
    plt.figure(figsize=(15, 5), dpi=200)
    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(img, 'gray')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title("OpenCV")
    plt.imshow(img_cv)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title("From ground")
    plt.imshow(img_c)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# %%
