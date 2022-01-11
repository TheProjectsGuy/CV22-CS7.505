# Create video from images in a given folder
"""
    Given images (numerically sorted 1 through N) in a folder, read 
    them and create a video (no audio, only video)
"""

# %% Import everthing
import cv2 as cv
import os
import glob
import argparse

# %% Argparse parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--imgs-folder', default="./seq", 
    help="Path to folder in which images are stored (all jpg files)")
parser.add_argument('-o', '--out-file', default="./out.avi", 
    help="Output file (uses AVI, XVID fourcc)")
parser.add_argument('-f', '--vid-fps', default=10.0, type=float,
    help="The desired FPS (as float) for the output file")

# %% Main entrypoint
if __name__ == "__main__":
    # Parse arguments
    args, unknown_args = parser.parse_known_args()
    
    # Variables
    imgs_path = f"{os.path.realpath(args.imgs_folder)}/*.jpg"
    fps = args.vid_fps
    out_file = args.out_file
    # File names (sorted numerically)
    img_fnames = sorted(glob.glob(imgs_path), key=len)
    shape_w_h = cv.imread(img_fnames[0]).shape[-2::-1]  # (W, H) of video
    # Video writer
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out_fhdlr = cv.VideoWriter(out_file, fourcc, fps, shape_w_h)

    # For ever image found, write it to the video file
    for i, img_file in enumerate(img_fnames):
        # Read file
        frame = cv.imread(img_file)
        if frame is None:
            print(f"Unable to read '{img_file}', skipping it!")
            continue
        # Write the image to video writer
        out_fhdlr.write(frame)
        # Preview
        cv.imshow("Video", frame)
        if cv.waitKey(int(1000/fps)) == ord('q'):
            print(f"Quit received after {i} frames")
            break
    # Cleanup
    out_fhdlr.release()
    cv.destroyAllWindows()

# %%
