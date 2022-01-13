# Apply background based chroma keying
"""
    Given a foreground video with a greenscreen in background, a video
    that should be underlayed as new background, and a green value (
    along with hue threshold). Use '-h' option to know more.

    More information at [1]

    [1]: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
"""

# %% Import everything
import cv2 as cv
import numpy as np
import argparse
import os

# %% Argparse variables
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Use 'q' to quit and 'c' to capture")
parser.add_argument('-g', '--g-val', default=216, 
    help="Green value (0-255)")
parser.add_argument('-t', '--hsv-thresh', default=5, type=float,
    help="Threshold percentage for hue (0-100)")
parser.add_argument('-w', '--img-fg', default="./videos/tiger-gs.mp4",
    help="Path to the foreground image")
parser.add_argument('-b', '--img-bg', 
    default="./videos/waterfall.mp4",
    help="Path to the background image")
parser.add_argument('-o', '--out-folder', default="./chroma_out", 
    help="Folder name (to store results). Must be exiting.")
parser.add_argument('-r', '--out-fps', default=30, 
    help="FPS (writing)")

# %% Main entrypoint
if __name__ == "__main__":
    args, extra_args = parser.parse_known_args()
    g_val = args.g_val  # Green value
    h_thresh = args.hsv_thresh   # Threshold (percentage)
    # Foreground and background files
    fg_file = os.path.realpath(args.img_fg)
    bg_file = os.path.realpath(args.img_bg)
    # Output folder and FPS
    vid_out_file = os.path.realpath(args.out_folder)
    out_fps = int(args.out_fps)    # Output FPS
    # Convert green value to HSV
    h, s, v = cv.cvtColor(np.array([[[0, g_val, 0]]], dtype=np.uint8), 
        cv.COLOR_BGR2HSV)[0,0]   # Green -> HSV
    print(f"HSV: {h}, {s}, {v}")

    # Read a frame from both streams
    v1 = cv.VideoCapture(fg_file)
    ret, img_fg = v1.read()
    v1.release()
    # Main task
    i = 0
    v1 = cv.VideoCapture(fg_file)
    v2 = cv.VideoCapture(bg_file)
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    vwrite = cv.VideoWriter(f"{vid_out_file}/chroma_res.avi", fourcc,
        out_fps, img_fg.shape[-2::-1])
    while True:
        ret, img_fg = v1.read()
        if not ret:
            print("Foreground video finished")
            break
        ret, img_bg = v2.read()
        if not ret:
            print("Background video finished")
            break
        # Mask through HSV
        img_fg_hsv = cv.cvtColor(img_fg, cv.COLOR_BGR2HSV)
        lower_green = np.array([h*(1-h_thresh/100), 180, 180])
        upper_green = np.array([h*(1+h_thresh/100), 255, 255])
        # img_fg_hsv channels in range (AND of each range check)
        mask = cv.inRange(img_fg_hsv, lower_green, upper_green)
        # Construct final image
        img_final = img_fg.copy()
        img_final[mask == 255] = img_bg[mask == 255]
        # Show result
        cv.imshow("Fusion", img_final)
        vwrite.write(img_final)
        key = cv.waitKey(int(1000/30))
        if key == ord('q'):
            print("Exit signal received")
            break
        elif key == ord('c'):
            i += 1  # New image
            cv.imwrite(f"{vid_out_file}/mask{i}.jpg", mask)
            cv.imwrite(f"{vid_out_file}/img_fg{i}.jpg", img_fg)
            cv.imwrite(f"{vid_out_file}/img_bg{i}.jpg", img_bg)
            cv.imwrite(f"{vid_out_file}/img_f{i}.jpg", img_final)
            print(f"Saved image '{i}' in folder '{vid_out_file}'")
    v1.release()
    v2.release()
    vwrite.release()
    cv.destroyAllWindows()
    print("Video writing completed")

# %%
