# A python script to rename images in a file
"""
    Give a folder, with images in some order, the result will be
    another folder with images named 1 through N. The folders must
    already be existing.
"""

# %% Import everything
import os
import cv2 as cv
import glob

# %% Properties
folder_in = "./imgseq"
folder_out = "./seq"

# %% Some processing
f = glob.glob(f"{folder_in}/*.jpg") # All raw files
f1 = [os.path.basename(k) for k in f]   # File names only
ns = [k[k.find("(")+1:k.find(")")] for k in f1] # Get index numbers

# %% Read and save
for i, img in enumerate(f):
    frame = cv.imread(img)
    if frame is None:
        print(f"File '{img}' could not be read")
        continue
    cv.imwrite(f"{folder_out}/{ns[i]}.jpg", frame)

# %%
