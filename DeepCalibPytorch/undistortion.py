import numpy as np
import torch
import cv2
from PIL import Image
from torchvision.transforms import v2
from undistSphIm import undistSphIm
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    classes_focal = list(np.arange(50, 500 + 1, 10))
    classes_distortion = list(np.arange(0, 90 + 1, 4) / 100.)
    device = 'cuda:1'
    model_path = "model_80.pth"
    image_path = "distortion_images/pano_056327084ab7d7c4a25500f131bf442f_f_130_d_0.84.jpg"
    model = torch.load(model_path).eval().to(device)
    image = cv2.imread(image_path)

    preprocess = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    height, width, _ = image.shape
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
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

        Image_und = undistSphIm(image, Paramsd, Paramsund)
        cv2.imwrite(f"output_undistorted.png", Image_und)