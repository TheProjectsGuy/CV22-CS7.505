# %% Import everything
import sympy as sp
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

# %% Read image and get the center pixel
# Function to select a pixel
# A function that takes an image and returns the marked indices
def get_clicked_points(img, img_winname = "Point Picker", 
    dmesg = False):
    """
        Get the clicked locations as a list of [x, y] points on image
        given as 'img'. Note that the origin is at the top left corner
        with X to the right and Y downwards.

        Parameters:
        - img: np.ndarray   shape: N, M, C
            An image, should be handled by OpenCV or be a numpy array
            (height, width, channels). The passed image is not altered
            by the function.
        - img_winname: str  default: "Point Picker"        
            Window name (to be used by OpenCV)
        - dmesg: bool or str    default: False
            If True (or `str` type) a string for debug is printed. If
            the type is `str`, then the string is prepended to the
            debug message.

        Returns:
        - img_points: list
            A list of [x, y] points clicked on the image
        - _img: np.ndarray  shape: N, M, C
            The same image, but annotated with points clicked. Random
            colors are assigned to each point.
    """
    img_points = [] # A list of [x, y] points (clicked points)
    _img: np.ndarray = img.copy()   # Don't alter img

    def img_win_event(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONUP:
            # Print the debug message (if True or 'str')
            if dmesg == True or type(dmesg) == str:
                db_msg = f"Clicked on point (x, y): {x}, {y}"
                if type(dmesg) == str:
                    db_msg = dmesg + db_msg
                print(db_msg)
            # Record point
            img_points.append([x, y])   # Record observation
            # -- Put marker on _img for the point --
            # Random OpenCV BGR color as tuple
            _col = tuple(map(int, np.random.randint(0, 255, 3)))
            # Add circle
            cv.circle(_img, (x, y), 10, _col, -1)
            # Add text
            cv.putText(_img, f"{len(img_points)}", (x, y-15),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, _col, 2, cv.LINE_AA)
    
    # Create GUI Window
    cv.namedWindow(img_winname, cv.WINDOW_NORMAL)
    cv.resizeWindow(img_winname, 1242, 375)  # Window (width, height)
    cv.setMouseCallback(img_winname, img_win_event)
    # Main loop
    while True:
        cv.imshow(img_winname, _img)
        k = cv.waitKey(1)
        if k == ord('q'):
            break
    cv.destroyWindow(img_winname)
    # Return results
    return img_points, _img


img_location = "./image.png"
img_location = os.path.realpath(os.path.expanduser(img_location))
assert os.path.isfile(img_location)
car_img = cv.imread(img_location)
# Click center pixel
cent_px, _ = get_clicked_points(car_img, "Pick Centroid Pixel")
cpx, cpy = cent_px[0]   # Resolve as pixel values
print(f"Center pixel is: {cpx}, {cpy}")

# %% Symbols
# -- Known parameters (as floats) --
vl_val, vw_val, vh_val = 4.10, 1.51, 1.38   # L, W, H in m
vth_val = np.deg2rad(5) # Angle (in rad)
ch_val = 1.65   # Cam height in m
# vcx_val, vcy_val = 839, 234 # Camera pixel of vehicle center
vcx_val, vcy_val = cpx, cpy # Camera pixel of vehicle center
K_val = [   # Camera intrinsic parameter matrix
    [7.2153e+02,0,6.0955e+02],
    [0,7.2153e+02,1.7285e+02],
    [0,0,1]]
K_np = np.array(K_val, float)   # As numpy
# - The above will only be used in the end -

# -- Known Parameters (as symbols) --
# Vehicle properties
vl, vw, vh = sp.symbols(r"V_l, V_w, V_h")   # dimensions (L, W, H)
vth = sp.symbols(r"V_\theta")   # Z rotation for vehicle (in rad)
# Camera properties
ch = sp.symbols(r"C_h") # Camera height (from ground)
# Camera projection matrix
K_11, K_12, K_13 = sp.symbols(r"k_{11}, k_{12}, k_{13}")
K_22, K_23, K_33 = sp.symbols(r"k_{22}, k_{23}, k_{33}")
K_sp = sp.Matrix([[K_11, K_12, K_13], [0, K_22, K_23], [0, 0, K_33]])
vcx, vcy = sp.symbols(r"V'_{c_x}, V'_{c_y}")    # Pixel of car center

# -- Unknown parameters --
# Vehicle parameters
vx, vy = sp.symbols(r"V_x, V_y")    # Vehicle X and Y from {world}

# %% Prior to main work
# Image point (homogeneous coordinates)
vimg = sp.Matrix([vcx, vcy, 1])
# -- Homogeneous Transformations --
# - TF {vehicle} in {world} -
# Rotation for {vehicle} in {world}
R_w_v = sp.Matrix([ # Rot(Z, vth)
    [sp.cos(vth), -sp.sin(vth), 0],
    [sp.sin(vth), sp.cos(vth), 0],
    [0, 0, 1]])
# Vehicle origin (in {world} - homogeneous coordinates)
vorg_w = sp.Matrix([vx, vy, 0, 1])
# Homogeneous Transformation matrix ({vehicle} in {world})
tf_w_v = sp.Matrix.hstack(  # Stacking R_w_v and vorg_w
    sp.Matrix.vstack(R_w_v, sp.Matrix([[0, 0, 0]])), vorg_w)
# - TF {camera} in {world} -
# Rotation from world to camera
R_w_c = sp.Matrix([ # Z out of cam, Y down, X to right
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]])
# Camera origin (in {world} - homogeneous coordinates)
corg_w = sp.Matrix([0, 0, ch, 1])
# Homogeneous Transformation matrix ({camera} in {world})
tf_w_c = sp.Matrix.hstack(  # Stacking R_w_v and vorg_w
    sp.Matrix.vstack(R_w_c, sp.Matrix([[0, 0, 0]])), corg_w)
# - TF {world} in {camera} -
tf_c_w = sp.Matrix.hstack(
    sp.Matrix.vstack(R_w_c.T, sp.Matrix([[0, 0, 0]])), 
    sp.Matrix.vstack(
        -R_w_c.T * sp.Matrix(corg_w[0:3]), sp.Matrix([[1]]))
    )   # Invert the transformation matrix

# %% Equation for resolving points
# Vehicle center in {vehicle}
vc_v = sp.Matrix([vl/2, vw/2, vh/2, 1])
# Vehicle center in {world}
vc_w = tf_w_v * vc_v
# Vehicle center in {camera}
vc_c = tf_c_w * vc_w

# %% Camera projection equations (Main solution)
lhs_eq = K_sp.inv() * vimg  # Image projected to the world
rhs_eq = sp.Matrix([    # Vehicle center in camera frame [X;Y;Z]
    [vc_c[0]/vc_c[3]],
    [vc_c[1]/vc_c[3]],
    [vc_c[2]/vc_c[3]]])
# The last value of LHS is 1 (projection), set the same of for RHS
rhs_eqn = rhs_eq / rhs_eq[2] # Last value corresponds
lhs_eqn = lhs_eq / lhs_eq[2] # Last value corresponds
eq_s = sp.Eq(lhs_eqn, rhs_eqn)    # Equality to solve
sols = sp.solvers.solve(eq_s, [vx, vy]) # Solutions to the equality
vx_sol = sols[vx]
vy_sol = sols[vy]

# %% Solution for vehicle positions
# Substitution values
val_subs = {
    vl: vl_val,
    vw: vw_val,
    vh: vh_val,
    ch: ch_val,
    vth: vth_val,
    vcx: vcx_val,
    vcy: vcy_val,
    K_11: K_np[0, 0], K_12: K_np[0, 1], K_13: K_np[0, 2],
    K_22: K_np[1, 1], K_23: K_np[1, 2], K_33: K_np[2, 2]
}
vx_res = float(vx_sol.subs(val_subs))
vy_res = float(vy_sol.subs(val_subs))
print(f"Vehicle BRD at (X, Y): {vx_res:.4f}, {vy_res:.4f}")

# %% Show car with center pixel
car_img_plt = cv.cvtColor(car_img.copy(), cv.COLOR_BGR2RGB)
plt.figure(figsize=(15, 10))
plt.imshow(car_img_plt)
plt.title("Car with center pixel")
plt.plot(vcx_val, vcy_val, 'rx')
plt.savefig("./fig1.png")
plt.show()

# %%
# Transform solution (for vehicle to camera frame) as floats
vx_w_sol = vx_res
vy_w_sol = vy_res
tf_c_v_sp = tf_c_w * tf_w_v
tf_c_v = tf_c_v_sp.subs(val_subs).subs({vx: vx_w_sol, vy: vy_w_sol})
tf_c_v = np.array(tf_c_v, float)    # As numpy floats
print(f"Transformation from vehicle to camera frame is: \n{tf_c_v}")
# Camera projection matrix (in numpy)
K_np = np.array(K_sp.subs(val_subs), float)
print(f"Camera projection matrix is: \n{K_np}")

# %% Project the 3D points to the camera frame
points_v = np.array([   # Points in [X, Y, Z], in {Vehicle}
    [0, 0, 0],  # V_BRD
    [vl_val, 0, 0], # V_FRD
    [vl_val, vw_val, 0],    # V_FLD
    [0, vw_val, 0],     # V_BLD
    [0, 0, vh_val],     # V_BRU
    [vl_val, 0, vh_val],    # V_FRU
    [vl_val, vw_val, vh_val],   # V_FLU
    [0, vw_val, vh_val]     # V_BLU
])
# Convert to homogeneous coordinates
corners_v = np.vstack((points_v.T, np.ones((1, points_v.shape[0]))))
# Corners in camera frame
corners_c = tf_c_v @ corners_v
corners_c = corners_c[0:3, :]   # Loose the last row
# Convert to the camera coordinates
corners_img = K_np @ corners_c
# Scale to 1 (for pixel representations)
corners_img_px = corners_img / corners_img[2]

# %% Show bounding box
# Show the results
c_img = corners_img_px.astype(int)[0:2, :].T
plt.figure(figsize=(20, 10))
plt.imshow(car_img_plt)
plt.title("Bounding Box")
# Center
plt.plot(vcx_val, vcy_val, 'co')
# All bounding boxes
plt.plot(c_img[:, 0], c_img[:, 1], 'r.')
# Make lines
plt.plot(c_img[[0, 1, 2], 0], c_img[[0, 1, 2], 1], 'r--')
plt.plot(c_img[[2, 3, 0], 0], c_img[[2, 3, 0], 1], 'r-')
plt.plot(c_img[4:8, 0], c_img[4:8, 1], 'r-')
plt.plot(c_img[[7, 4, 0], 0], c_img[[7, 4, 0], 1], 'r-')
plt.plot(c_img[[1, 5], 0], c_img[[1, 5], 1], 'r--')
plt.plot(c_img[[2, 6], 0], c_img[[2, 6], 1], 'r-')
plt.plot(c_img[[3, 7], 0], c_img[[3, 7], 1], 'r-')
# Dotted diagonal lines
plt.plot(c_img[[0, 6], 0], c_img[[0, 6], 1], 'c:')
plt.plot(c_img[[1, 7], 0], c_img[[1, 7], 1], 'c:')
plt.plot(c_img[[2, 4], 0], c_img[[2, 4], 1], 'c:')
plt.plot(c_img[[3, 5], 0], c_img[[3, 5], 1], 'c:')
plt.savefig("./fig2.png")
plt.show()

# %% Rear axle part
# Rear axle in vehicle frame
rear_axle_v = np.array([0.2*vl_val, 0.5*vw_val, 0, 1])  # Homogeneous
# Rear axle in camera frame
rear_axle_c = tf_c_v @ rear_axle_v
raxle_c = rear_axle_c[:3]
print(f"Rear axle in camera frame is (X, Y, Z): {raxle_c}")
# Image point
raxle_img = K_np @ raxle_c
rax_i = raxle_img / raxle_img[2]
rax_i = rax_i.astype(int)
print(f"Rear axle in image at (x, y): {rax_i[0]}, {rax_i[1]}")

# %% Show bounding box with rear axle
# Show the results (with axle)
plt.figure(figsize=(20, 10))
plt.imshow(car_img_plt)
plt.title("Bounding Box with Axle")
# Center
plt.plot(vcx_val, vcy_val, 'co')
# All bounding boxes
plt.plot(c_img[:, 0], c_img[:, 1], 'r.')
# Make lines
plt.plot(c_img[[0, 1, 2], 0], c_img[[0, 1, 2], 1], 'r--')
plt.plot(c_img[[2, 3, 0], 0], c_img[[2, 3, 0], 1], 'r-')
plt.plot(c_img[4:8, 0], c_img[4:8, 1], 'r-')
plt.plot(c_img[[7, 4, 0], 0], c_img[[7, 4, 0], 1], 'r-')
plt.plot(c_img[[1, 5], 0], c_img[[1, 5], 1], 'r--')
plt.plot(c_img[[2, 6], 0], c_img[[2, 6], 1], 'r-')
plt.plot(c_img[[3, 7], 0], c_img[[3, 7], 1], 'r-')
# Dotted diagonal lines
plt.plot(c_img[[0, 6], 0], c_img[[0, 6], 1], 'c:')
plt.plot(c_img[[1, 7], 0], c_img[[1, 7], 1], 'c:')
plt.plot(c_img[[2, 4], 0], c_img[[2, 4], 1], 'c:')
plt.plot(c_img[[3, 5], 0], c_img[[3, 5], 1], 'c:')
# Rear axle
plt.plot(rax_i[0], rax_i[1], 'go')
plt.savefig("./fig3.png")
plt.show()

# %%
