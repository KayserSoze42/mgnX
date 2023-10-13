import os

import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import cv2


class LWM1Dataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.metadata = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        img_path = os.path.join(self.root_dir, self.metadata.iloc[idx, 0])
        image = cv2.imread(img_path)

        # xtrct

        mclass = self.metadata.iloc[idx, 1] # ed, ned or ded
        genus = self.metadata.iloc[idx, 2] # boletus, amanita
        species = self.metadata.iloc[idx, 3] # namen

        geolocation = eval(self.metadata.iloc[idx, 4]) # frmt /= impl 
        date_time = pd.to_datetime(self.metadata.iloc[idx, 5]) # -||-

        if image is None:
            print(f"Error reading {img_path}. eskipah")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # gray to rgb++

        if len(image.shape == 2):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        sample = {

                'image': image,
                'class': mclass,
                'genus': genus,
                'species': species,
                'geolocation': geolocation,
                'date_time': date_time

        }

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

data_transforms = transforms.Compose([

    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])
