 
import os
import numpy as np
import cv2  # OpenCV library
import time
from undistSphIm import undistSphIm

# Create the output directory if it doesn't exist
dist_folder = './undistion_result'
if not os.path.exists(dist_folder):
    os.makedirs(dist_folder)

# Read the contents of the prediction file
file_path = './prediction.txt'
with open(file_path, 'r') as file:
    file_content = file.readlines()
    

paths = [line.strip().split()[0] for line in file_content]
focal = [float(line.strip().split()[1]) for line in file_content]
distortion = [float(line.strip().split()[2]) for line in file_content]


# print("path:", paths)
# print("focal:", focal)
# print("dist:", distortion)

f = 0
dist = 0

for i in range(len(paths)):
    Idis = cv2.imread(paths[i])

    xi = distortion[i]  # distortion
    dist += xi
    ImH, ImW = Idis.shape[:2]
    f_dist = focal[i] * (ImW / ImH) * (ImH / 299)  # focal length
    f += f_dist
    u0_dist = ImW / 2
    v0_dist = ImH / 2

    Paramsd = {'f': f_dist, 'W': u0_dist * 2, 'H': v0_dist * 2, 'xi': xi}
    Paramsund = {'f': f_dist, 'W': u0_dist * 2, 'H': v0_dist * 2}

    start_time = time.time()
    Image_und = undistSphIm(Idis, Paramsd, Paramsund)
    end_time = time.time()
    print(f"Undistortion time: {end_time - start_time} seconds")

    paths_list = paths[i].split('/')
    res1 = paths_list[-1]
    res2 = res1.split('.')

    # out = int(res2[0])

    # filename = f'{out:04d}.jpg'
    filename = res2[0] + "_undistorted.jpg"
    fullname = os.path.join(dist_folder, filename)
    cv2.imwrite(fullname, Image_und)
