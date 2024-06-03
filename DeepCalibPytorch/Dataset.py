from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
import torch
import glob
import random

class ImageDataset(Dataset):
    def __init__(
        self, 
        image_dir,
        max_len,
        classes_focal, 
        classes_distortion,
        data_augmentation = True
    ):
        super().__init__()
        self.data_augmentation = data_augmentation
        self.classes_focal = classes_focal
        self.classes_distortion = classes_distortion
        filenames, focal_labels, distortion_labels = self.get_paths(image_dir, max_len)
        self.images = self.load_image(filenames)

        self.focal_labels = torch.tensor(focal_labels)
        self.distortion_labels = torch.tensor(distortion_labels)

        self.transform_none = v2.Compose([
            v2.Resize(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_augmentation = v2.Compose([
            v2.Resize(224),
            v2.RandomHorizontalFlip(p = 0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            CustomTransform(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.focal_labels)

    def __getitem__(self, idx):
        if self.data_augmentation == True:
            tensor = self.transform_augmentation(self.images[idx])
        else:
            tensor = self.transform_none(self.images[idx])
        return tensor, self.focal_labels[idx], self.distortion_labels[idx]
    
    def load_image(self, filenames):
        images = []
        for filename in filenames:
            image = Image.open(filename)
            images.append(image)
        return images

    def get_paths(self, IMAGE_FILE_PATH_DISTORTED, max_len):
        paths_train = glob.glob(IMAGE_FILE_PATH_DISTORTED + "*.jpg")
        parameters = []
        labels_focal_train = []
        for path in paths_train:
            curr_parameter = float((path.split('_f_'))[1].split('_d_')[0])
            parameters.append(curr_parameter)
            curr_class = self.classes_focal.index(curr_parameter)
            labels_focal_train.append(curr_class)
        labels_distortion_train = []
        for path in paths_train:
            curr_parameter = float((path.split('_d_'))[1].split('.jpg')[0])
            parameters.append(curr_parameter)
            curr_class = self.classes_distortion.index(curr_parameter)
            labels_distortion_train.append(curr_class)

        c = list(zip(paths_train, labels_focal_train, labels_distortion_train))
        random.shuffle(c)
        paths_train, labels_focal_train,labels_distortion_train = zip(*c)
        paths_train, labels_focal_train, labels_distortion_train = list(paths_train), list(labels_focal_train), list(labels_distortion_train)
        
        if max_len == -1:
            return paths_train, labels_focal_train, labels_distortion_train
        
        return paths_train[:max_len], labels_focal_train[:max_len], labels_distortion_train[:max_len]

class CustomTransform:
    def __init__(
        self
    ):
        self.std = 0.5
        self.contrast_low = 0.9
        self.contrast_high = 1.1
        self.brightness_low = -(5/255)
        self.brightness_high = (5/255)

    def __call__(self, tensor):
        tensor += torch.randn(tensor.size()) * self.std
        contrast = torch.FloatTensor(1).uniform_(self.contrast_low, self.contrast_high)
        brightness = torch.FloatTensor(1).uniform_(self.brightness_low, self.brightness_high)
        tensor = tensor * contrast + brightness
        return tensor

    def __repr__(self):
        return self.__class__.__name__