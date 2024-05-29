 
import numpy as np
import cv2

def undistSphIm(Idis, Paramsd, Paramsund):
    Paramsund['W'] = Paramsd['W'] * 3  # size of output (undist)
    Paramsund['H'] = Paramsd['H'] * 3

    # Parameters of the camera to generate
    f_dist = Paramsd['f']
    u0_dist = Paramsd['W'] / 2
    v0_dist = Paramsd['H'] / 2

    f_undist = Paramsund['f']
    u0_undist = Paramsund['W'] / 2
    v0_undist = Paramsund['H'] / 2
    xi = Paramsd['xi']  # distortion parameters (spherical model)

    # 1. Projection on the image
    grid_x, grid_y = np.meshgrid(np.arange(1, Paramsund['W']+1), np.arange(1, Paramsund['H']+1))
    X_Cam = grid_x / f_undist - u0_undist / f_undist
    Y_Cam = grid_y / f_undist - v0_undist / f_undist
    Z_Cam = np.ones((int(Paramsund['H']), int(Paramsund['W'])))

    # 2. Image to sphere cart
    xi1 = 0
    alpha_cam = (xi1 * Z_Cam + np.sqrt(Z_Cam**2 + (1 - xi1**2) * (X_Cam**2 + Y_Cam**2))) / (X_Cam**2 + Y_Cam**2 + Z_Cam**2)
    X_Sph = X_Cam * alpha_cam
    Y_Sph = Y_Cam * alpha_cam
    Z_Sph = Z_Cam * alpha_cam - xi1

    # 3. Reprojection on distorted
    den = xi * np.sqrt(X_Sph**2 + Y_Sph**2 + Z_Sph**2) + Z_Sph
    X_d = (X_Sph * f_dist / den) + u0_dist
    Y_d = (Y_Sph * f_dist / den) + v0_dist

    # 4. Final step interpolation and mapping
    Image_und = np.zeros((int(Paramsund['H']), int(Paramsund['W']), 3))

    for c in range(3):
        Image_und[:, :, c] = cv2.remap(Idis[:, :, c], X_d.astype(np.float32), Y_d.astype(np.float32), interpolation=cv2.INTER_CUBIC)

    return Image_und
