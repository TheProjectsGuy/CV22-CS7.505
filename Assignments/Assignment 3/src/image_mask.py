# Generate images through mask
"""
    Given a mask and an image, generate the masked image. Also contains
    tests for some metrics
"""

# %%
import numpy as np
import cv2 as cv

# %% Properties
img = cv.imread("../images/banana1.jpg")
img_mask = cv.imread("./mask_out.bmp", cv.IMREAD_GRAYSCALE)
img_gt_mask = cv.imread("./../ground_truth/banana1.bmp", cv.IMREAD_GRAYSCALE)

# %% Show the masked images
img_masked = cv.bitwise_and(img, img, mask=img_mask)
img_gt_masked = cv.bitwise_and(img, img, mask=img_gt_mask)

# %% Show masked images
cv.imshow("Masked Image", img_masked)
cv.imshow("GT Masked Image", img_gt_masked)
cv.waitKey(0)
cv.destroyAllWindows()

# %% Accuracy
incorr_label_img = cv.bitwise_xor(img_mask, img_gt_mask)    # Wrong labels
num_pix = img_mask.shape[0] * img_mask.shape[1] # Total pixels
acc = 1 - (np.count_nonzero(incorr_label_img)/num_pix)
print(f"Accuracy is: {acc*100:.3f} %")

# %%
corr_label_img = cv.bitwise_not(incorr_label_img)

# %% Jaccard Similarity
intersect_img = cv.bitwise_and(img_mask, img_gt_mask)
union_img = cv.bitwise_or(img_mask, img_gt_mask)
jacc_sim = np.count_nonzero(intersect_img)/np.count_nonzero(union_img)
print(f"Jaccard similarity: {jacc_sim*100:.3f} %")

# %% Dice similarity coefficient
nume = 2 * np.count_nonzero(cv.bitwise_and(img_mask, img_gt_mask))
deno = np.count_nonzero(img_mask) + np.count_nonzero(img_gt_mask)
dice_f1 = nume/deno
print(f"Dice similarity: {dice_f1*100:.3f} %")

# %%
cv.imshow("Temp", corr_label_img)
cv.waitKey(0)
cv.destroyAllWindows()

# %%
