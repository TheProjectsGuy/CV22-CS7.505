# Change the color channels of an image
"""
    Flips the Reg and Blue channel of an image
"""

# %% Import everything
import cv2 as cv
import numpy as np

# %% Functions
# Show image in a window
def show_img(img, win_name = "Image"):
    cv.namedWindow(win_name, cv.WINDOW_GUI_EXPANDED)
    while True:
        cv.imshow(win_name, img)
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyWindow(win_name)

# %% Main module
if __name__ == "__main__":
    # Read image
    img_in = cv.imread("./images/test.jpg")
    # Show image
    show_img(img_in, "Input")
    img_out = img_in[:,:,::-1]
    show_img(img_out, "Output")
    # Save output
    cv.imwrite("./images/output.jpg", img_out)

# %%
