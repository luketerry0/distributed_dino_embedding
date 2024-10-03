from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch


class StreetImageDataset(Dataset):
    def __init__(self, imageRootDir, transform=None):

        self.image_filter = lambda filename: filename.endswith(".jpg") or filename.endswith(".jpeg")

        length = 0
        for root, dirs, files in os.walk(imageRootDir):
            #pics = [x for x in files if self.image_filter(x)]
            length += len(files)

        self.imageRootDir = imageRootDir
        self.transform = transform
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # return a specific image from the filesystem
        for root, dirs, files in os.walk(self.imageRootDir):
            # pics = [x for x in files if self.image_filter(x)]
            idx -= len(files)
            if (idx <= 0):
                idx += len(files)
                file = root + '/' + files[idx]
                break
        
        # read in the file as a PIL image
        image = Image.open(file)
        if self.transform:
            image = self.transform(image)
        return image, file
