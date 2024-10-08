from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd

from torchvision.transforms import v2
import torchvision.transforms as transforms
import torchvision

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch


class StreetImageDataset(Dataset):
    def __init__(self, imageRootDir, transform=None):

        self.image_filter = lambda filename: filename.endswith(".jpg") or filename.endswith(".jpeg")

        length = 0
        for root, dirs, files in os.walk(imageRootDir):
            pics = [x for x in files if self.image_filter(x)]
            length += len(pics)

        self.imageRootDir = imageRootDir
        self.transform = transform
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # return a specific image from the filesystem
        for root, dirs, files in os.walk(self.imageRootDir):
            pics = [x for x in files if self.image_filter(x)]
            idx -= len(pics)
            if (idx < 0):
                idx += len(pics)
                file = root + '/' + pics[idx]
                break
        
        # read in the file as a PIL image
        image = Image.open(file)
        if self.transform:
            image = self.transform(image)
        return image, file

if __name__ == "__main__":

    transform = v2.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        transforms.Resize(size=(256, 256), antialias=True),
                        transforms.CenterCrop((224, 224)),
                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                            ),
                    ]
                    )

    dataset = StreetImageDataset("./test_bronx", transform=transform)
    data = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1)


    image, file = next(iter(data))
    print(image.nelement()*image.element_size())
    print(file)