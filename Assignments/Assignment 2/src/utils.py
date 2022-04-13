# Utility functions for the assignment
"""
    Some helper functions
"""

# %% Import everything
import numpy as np
import cv2 as cv

# %% Functions

# %% Rescaling functions
# Rescale to uint8 (useful for displaying images)
def imrescale_uint8(img: np.ndarray):
    """
        Scales the passed image 'img' in the range 0 to 255. Useful
        for previewing images.
    """
    img_r = ((img - img.min())/img.ptp()) * 255 + 0
    img_r = np.array(img_r, np.uint8)
    return img_r

# Rescale to unit (0 to 1)
def imrescale_unit(img: np.ndarray):
    """
        Scales an image to [0, 1] range (as float)
    """
    img_r = ((img - img.min())/img.ptp())
    img_r = np.array(img_r, float)
    return img_r

# %%
