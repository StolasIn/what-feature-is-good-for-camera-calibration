import numpy as np
import torch
import cv2
from torchvision.transforms import v2
import glob
from undistSphIm import undistSphIm
import warnings
warnings.filterwarnings("ignore")

def read_images(folder):
    filenames = glob.glob(f"{folder}/*.jpg")
    filenames.sort()
    images = []
    image_names = []
    for filename in filenames:
        image_names.append(filename.replace(".jpg", "").replace(f"{folder}/", ""))
        image = cv2.imread(filename)
        images.append(image)
    return image_names, images

if __name__ == '__main__':
    classes_focal = list(np.arange(50, 500 + 1, 10))
    classes_distortion = list(np.arange(0, 90 + 1, 4) / 100.)
    device = 'cuda:1'
    model_path = "model_50.pth"
    folder_path = "distortion_images"
    names, images = read_images(folder_path)
    model = torch.load(model_path).eval().to(device)

    preprocess = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for i in range(len(images)):
            height, width, _ = images[i].shape
            tensor = preprocess(images[i]).unsqueeze(0)
            tensor = tensor.to(device)
            focal, distortion = model(tensor)
            focal_class = torch.argmax(focal, dim = 1)
            distortion_class = torch.argmax(distortion, dim = 1)
            focal_parameters = classes_focal[focal_class]
            distortion_parameters = classes_distortion[distortion_class]
            print(focal_parameters, distortion_parameters) 

            f_dist = focal_parameters * (width / height) * (height / 299)  # focal length

            Paramsd = {'f': f_dist, 'W': width, 'H': height, 'xi': distortion_parameters}
            Paramsund = {'f': f_dist, 'W': width, 'H': height}

            Image_und = undistSphIm(images[i], Paramsd, Paramsund)
            cv2.imwrite(f"undistortion_images/{names[i]}_undistorted.png", Image_und)