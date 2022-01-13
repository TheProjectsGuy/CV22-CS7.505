# Capture images from a webcam
"""
    Given a webcam index and a destination folder (already existing),
    preview the feed and capture images using it. Press 'c' to capture
    an image and 'q' to quit.
"""

# %% Import everything
import cv2 as cv
import os
import argparse

# %% Argument parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Press 'c' to capture, 'q' to quit")
parser.add_argument('-c', '--cam', default=0, type=int, 
    help="Camera index")
parser.add_argument('-o', '--out-folder', default="./camimgs", 
    help="Output folder (should already exist)")

# %% Main entrypoint
if __name__ == "__main__":
    args, extra_args = parser.parse_known_args()
    webcam_id = args.cam
    out_folder = args.out_folder

    # Main camera work
    cam = cv.VideoCapture(webcam_id)
    imc = 0 # Image counter
    while True:
        ret, img = cam.read()
        if not ret:
            print("Camera did not give frames")
            break
        cv.imshow("Feed", img)
        key = cv.waitKey(1)
        if key == ord('q'):
            print("Quit command received")
            break
        elif key == ord('c'):
            imc += 1
            fname = f"{out_folder}/{imc}.jpg"
            print(f"Capturing image to '{fname}'")
            cv.imwrite(fname, img)
    cam.release()
    cv.destroyAllWindows()

# %%
