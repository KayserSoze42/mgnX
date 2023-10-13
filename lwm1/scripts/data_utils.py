import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import cv2

class LWM1Dataset(Dataset):
    
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(self.root_dir)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        img_path = os.path.join(self.root_dir, self.image_list[idx])
        image = cv2.imread(img_path)

        if image is None:
            print(f"Error reading {img_path}. eskipah")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # gray to rgb++

        if len(image.shape == 2):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            image = self.transform(image)

        return image

data_transforms = transforms.Compose([

    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])
