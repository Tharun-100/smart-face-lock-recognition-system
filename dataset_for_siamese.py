import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageNameDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_names = os.listdir(folder_path)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.folder_path, image_name)
        image = Image.open(image_path).convert('L')  # Open image and convert to gray scale:
        
        if self.transform is not None:
            image = self.transform(image)
        # image_name=image_name-'.jpg' this is not allowed
        image_name=image_name[:-4]
        return image, image_name

    def __len__(self):
        return len(self.image_names)
