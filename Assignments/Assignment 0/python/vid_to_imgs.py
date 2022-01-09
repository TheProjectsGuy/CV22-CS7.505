# Convert a video to images
"""
    Given a video, generate the images. Run as main.
"""

# %% Import everything
import cv2 as cv
import numpy as np
import argparse
import sys
import os

# %% Argument parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--vid-file', default="./videos/vtest.avi",
    help="Video file to read")
parser.add_argument('-n', '--num-imgs', default=0, type=int,
    help="The maximum number of images to output from video (0=all)")
parser.add_argument('-o', '--out-prefix', default="./out/img",
    type=str, help="Output prefix for images")

# %% Main entrypoint
if __name__ == "__main__":
    # Parse all (known) arguments
    args, unknown_args = parser.parse_known_args(sys.argv)
    # Check if output directory (if passed) exists
    out_dir = os.path.split(args.out_prefix)[0]
    if not os.path.isdir(out_dir):
        print(f"Folder {out_dir} is being created")
        os.makedirs(out_dir)
    # Read video from file
    cap = cv.VideoCapture(args.vid_file)
    try:
        # Read frames
        fnum = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Probably EOF reached!")
                break
            cv.imshow("Video Feed", frame)
            if cv.waitKey(100) == ord('q'):
                print(f"Break encountered")
                break
            if args.num_imgs == 0 or fnum < args.num_imgs:
                # Write to disk
                cv.imwrite(f"{args.out_prefix}{fnum+1}.jpg", frame)
                fnum += 1
            else:
                print(f"Reached {fnum} frames")
                break
        print(f"Wrote {fnum} frames under {args.out_prefix}*.jpg")
    finally:
        # Cleanup
        cap.release()
        cv.destroyAllWindows()

# %%
