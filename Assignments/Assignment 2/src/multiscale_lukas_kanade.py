# Multi-scale Lukas-Kanade
"""
    Contains the functions for refining the optical flow and doing
    multi-scale calculations of optical flow (in an image pyramid).
"""

# %% Import everything
import numpy as np
import cv2 as cv
from lukas_kanade import lukas_kanade_dense, lukas_kanade_grid
from matplotlib import pyplot as plt


# %%
# Refine optical flow
def opt_flow_refine_grid(img_f, img_g, win_h, win_w, popt_flow, 
        sf_val = 15, ev_thresh = 1.0):
    """
        Refine the optical flow using priors. Uses grid technique (no
        dense correspondence).

        Parameters:
        - img_f: np.ndarray         shape: H, W or H, W, 3
            Image 1 (initial). Could be color or grayscale.
        - img_g: np.ndarray         shape: H, W or H, W, 3
            Image 2 (final). Could be color or grayscale.
        - win_h: int
            The height of window
        - win_w: int
            The width of window
        - popt_flow: tuple
            A tuple of position_np, opt_flow_np (each of shape: N, 2)
            containing the prior optical flow
        - sf_val: float
            The scaling factor for optical flow
        - ev_thresh: float
            The threshold for grid optical flow
        
        Returns:
        - nflow: np.ndarray     shape: N, 2
            New optical flow at the same positions
    """
    # Generate modified image
    def get_mod_img(img1, positions_np, optical_flow_np, sf_val, 
            win_h, win_w):
        # Nearest integer value using sf_val
        opt_flow_filtered = np.rint(optical_flow_np * sf_val)\
            .astype(int)
        # Positions are Height, Width (Y, X)
        pos_filtered_np = positions_np + opt_flow_filtered[:, [1, 0]]
        # All positions are height, width (Y, X)
        img_modified = img1.copy()
        imH, imW, imC = img1.shape
        for i in range(len(pos_filtered_np)):
            # Source patch: Guaranteed to exist in bounds
            spxmin = positions_np[i, 1] - win_w // 2
            spxmax = positions_np[i, 1] + win_w // 2
            spymin = positions_np[i, 0] - win_h // 2
            spymax = positions_np[i, 0] + win_h // 2
            # Destination patch
            dpxmin = pos_filtered_np[i, 1] - win_w // 2
            dpxmax = pos_filtered_np[i, 1] + win_w // 2
            dpymin = pos_filtered_np[i, 0] - win_h // 2
            dpymax = pos_filtered_np[i, 0] + win_h // 2
            # Shifts in Window (for source)
            sxmin, sxmax, symin, symax = 0, 0, 0, 0
            # Destination could be out of bounds
            if dpxmin < 0:  # Start of destination in -ve X
                # print(f"X_start < 0 === {dpxmin} < 0")
                sxmin = 0 - dpxmin  # Reduce starting (+ve)
            if dpxmax > imW:    # End of destination in > imW (out)
                # print(f"X_end > imW === {dpxmax} > {imW}")
                sxmax = imW - dpxmax    # Reduce ending (-ve)
            if dpymin < 0:  # Start of destination in -ve Y
                # print(f"Y_start < 0 === {dpymin} < 0")
                symin = 0 - dpymin  # Reduce starting (+ve)
            if dpymax > imH:    # End of destination in > imH (out)
                # print(f"Y_end > imH === {dpymax} > {imH}")
                symax = imH - dpymax    # Reduce ending (-ve)
            dpxmin = dpxmin + sxmin
            dpxmax = dpxmax + sxmax
            dpymin = dpymin + symin
            dpymax = dpymax + symax
            img_modified[dpymin:dpymax, dpxmin:dpxmax] = \
                img1[spymin+symin:spymax+symax,
                spxmin+sxmin:spxmax+symax]
        # Return the modified image
        return img_modified
    # Extract prior optical flow
    pos_np, opt_flow_np = popt_flow
    # Get the modified image
    img_f_modified = get_mod_img(img_f, pos_np, opt_flow_np, sf_val, 
        win_h, win_w)
    # Run optical flow again
    imgf = img_f_modified.copy()
    if len(img_f.shape) == 3:
        imgf = cv.cvtColor(imgf, cv.COLOR_BGR2GRAY)
    imgg = img_g.copy()
    if len(img_g.shape) == 3:
        imgg = cv.cvtColor(imgg, cv.COLOR_BGR2GRAY)
    # New optical flow
    _, noflow_np, _ = lukas_kanade_grid(imgf, imgg, win_h, win_w, 
        ev_thresh)
    # Add with previous
    nflow = noflow_np + opt_flow_np # New + old
    return nflow


# %% Multi-scale
def opt_flow_multiscale_grid(img_f, img_g, num_levels, win_h, win_w, 
        sf_val = 7, ev_thresh = 0.05):
    # Interpolating optical flow maps
    def interp_opflow(old_img, new_img, old_flow):
        old_img = old_img.copy()
        new_img = new_img.copy()
        old_flow = old_flow.copy()
        # New optical flow
        flow1 = old_flow.reshape(old_img.shape[0]//win_h, 
            old_img.shape[1]//win_w, 2)     # Old optical flow
        dst_coords = np.zeros((new_img.shape[0]//win_h, 
            new_img.shape[1]//win_w, 2))    # New optical flow
        # Get the new optical flow
        for i in range(flow1.shape[0]):
            for j in range(flow1.shape[1]):
                # print(f"{i, j} -> {2*i, 2*j}")
                dst_coords[2*i, 2*j] = flow1[i, j]
        coords = []
        for i in range(new_img.shape[0]//win_h):
            for j in range(new_img.shape[1]//win_w):
                coords.append([i*win_h + win_h//2, 
                    j*win_w + win_w//2])
        coords_np = np.array(coords)
        new_flow = dst_coords.reshape(-1, 2)
        return coords_np, new_flow
    # Downscale images
    img1_s = cv.resize(img_f, None, None, 0.5**(num_levels-1), 
        0.5**(num_levels-1), cv.INTER_LINEAR)   # Image f (1)
    img2_s = cv.resize(img_g, None, None, 0.5**(num_levels-1), 
        0.5**(num_levels-1), cv.INTER_LINEAR)   # Image g (2)
    # Get optical flow
    coords, oflow, _ = lukas_kanade_grid(
        cv.cvtColor(img1_s.copy(), cv.COLOR_BGR2GRAY), 
        cv.cvtColor(img2_s.copy(), cv.COLOR_BGR2GRAY), win_h, win_w, 
        ev_thresh)
    # Refine the optical flow
    noflow = opt_flow_refine_grid(img1_s, img2_s, win_h, win_w, 
        (coords, oflow), sf_val, ev_thresh)
    # Multi-level processing
    for levels in range(num_levels-1, 0, -1):
        # New scale factor
        img1_ns = cv.resize(img_f, None, None, 0.5**(levels-1), 
            0.5**(levels-1), cv.INTER_LINEAR)   # New scaled image
        img2_ns = cv.resize(img_g, None, None, 0.5**(levels-1), 
            0.5**(levels-1), cv.INTER_LINEAR)
        # Interpolate the previous optical flow
        coords_np, oflow_np = interp_opflow(img1_s, img1_ns, noflow)
        # Refine optical flow for this level
        nflow = opt_flow_refine_grid(img1_ns, img2_ns, win_h, win_w, 
            (coords_np, oflow_np), sf_val, ev_thresh)
        # This is the new setting
        img1_s = img1_ns.copy()
        img2_s = img2_ns.copy()
        noflow = nflow.copy()
    out_coords = coords_np
    out_oflow = nflow
    return out_coords, out_oflow


# %%

# %%
if __name__ == "__main__":
    win_h, win_w = 15, 15
    sf_val = 7
    ev_thresh = 0.05
    img1 = cv.imread(
        "./../data/all-frames-colour/RubberWhale/frame07.png")
    img2 = cv.imread(
        "./../data/all-frames-colour/RubberWhale/frame08.png")
    positions_np, optical_flow_np, tvs = lukas_kanade_grid(
        cv.cvtColor(img1.copy(), cv.COLOR_BGR2GRAY), 
        cv.cvtColor(img2.copy(), cv.COLOR_BGR2GRAY), 
        win_h = win_h, win_w = win_w, ev_thresh = 0.05)
    plt.figure(figsize=(6, 9))
    plt.subplot(2,1,1)
    plt.title("X-hist")
    _ = plt.hist(sf_val * optical_flow_np[:, 0], 100)
    plt.subplot(2,1,2)
    plt.title("Y-hist")
    _ = plt.hist(sf_val * optical_flow_np[:, 1], 100)
    plt.show()

    noflow_np = opt_flow_refine_grid(img1, img2, win_h, win_w, 
        (positions_np, optical_flow_np), sf_val)
    err = np.sum(np.linalg.norm(optical_flow_np - noflow_np, axis=1))
    print(f"Error: {err}")

    out_coords, out_oflow = opt_flow_multiscale_grid(img1, img2, 3, 
        win_h, win_w, sf_val, ev_thresh)

    plt.title("Grid Optical Flow (multi-level)")
    plt.imshow(img1[..., ::-1])
    for pos, h in zip(out_coords, out_oflow):
        if np.allclose(h, np.zeros_like(h)):
            plt.plot(pos[1], pos[0], 'r.')
        else:
            plt.quiver(pos[1], pos[0], h[0], h[1], color='red', 
                width=5e-3, scale=3)
    plt.tight_layout()
    plt.show()

    positions_np, optical_flow_np, tvs = lukas_kanade_grid(
        cv.cvtColor(img1.copy(), cv.COLOR_BGR2GRAY), 
        cv.cvtColor(img2.copy(), cv.COLOR_BGR2GRAY), 
        win_h = win_h, win_w = win_w, ev_thresh = 0.05)
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


# %%

# %% Experimental Section
