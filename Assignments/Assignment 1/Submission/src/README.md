# Source files

The main source files for submission

## Table of contents

- [Source files](#source-files)
    - [Table of contents](#table-of-contents)
    - [Contents](#contents)
    - [Submission](#submission)

## Contents

The contents of this folder are described below

| S. No. | Item | Description |
| :---- | :----- | :----- |
| 1 | [Assignment01.ipynb](./Assignment01.ipynb) | The notebook submission |
| 2a | [clicked_corres.npz](./clicked_corres.npz) | Correspondences clicked for the given `black-dots` image |
| 2b | [clicked_corres_cam.npz](./clicked_corres_cam.npz) | Correspondences clicked for the `cam_1.jpg` image (personal camera capturing a known reference object in the world) |
| 2c | [P_dlt6.npy](./P_dlt6.npy), and [P_dlt_ransac.npy](./P_dlt_ransac.npy) | The camera projection matrix (for the `black-dots` camera) using DLT on 6 points and RANSAC (respectively) |
| 2d | [P_cam_dlt6.npy](./P_cam_dlt6.npy), and [P_cam_dlt_ransac.npy](./P_cam_dlt_ransac.npy) | The camera projection matrix for the personal camera using DLT on 6 points and RANSAC (respectively) |
| 3a | [cam_1.jpg](./cam_1.jpg) | Image of calibration object taken from a personal camera |
| 3b | [cam_cb_1.jpg](./cam_cb_1.jpg) through [cam_cb_8.jpg](./cam_cb_8.jpg) | Eight images of the checkerboard (the actual ones are in the `data` folder). A reference output is shown in [cb_out_8.jpg](./cb_out_8.jpg) |

## Submission

The submission has the following parts, answering the mentioned questions

1. `DLT - Direct Linear Transform`: Answers the questions 1 through 3
2. `Distortion & Wireframe`: Answers the questions 4 (using the given image and a sample from OpenCV repo), 5, and 7
3. `Zhang's Method`: Answers the question 6
4. `Theory questions`: Answers the question 8
5. `Personal Camera`: Answers the question 9 through 10

To save time and to avoid having to pick correspondences over and over again, the notebook reads them from the saved file. The images are also included.
