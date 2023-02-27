import torch
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import PIL
import os
import numpy as np
import random

class ImageDataset(Dataset):
    def __init__(self, folder, max_image_count = -1, img_size=64, preload=True):
        self.folder = folder
        self.files = os.listdir(folder)

        self.img_size = img_size

        if max_image_count != -1:
            self.files = random.sample(self.files, max_image_count)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])

        self.inverse_transform = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2),
            transforms.Lambda(lambda x: x.clamp(0, 1)),
            transforms.ToPILImage()
        ])


        self.preloaded_imgs = []
        if preload:
            if not os.path.exists("preprocessed"):
                os.mkdir("preprocessed")

            if os.path.exists(f"preprocessed/preloaded{max_image_count}_{self.img_size}.pt"):
                print("PRELOADING FROM FILE")
                self.preloaded_imgs = torch.load(f"preprocessed/preloaded{max_image_count}_{self.img_size}.pt")
                print("PRELOAD DONE")
            else:
                for i, file in enumerate(self.files):
                    print("PRELOADING", i, "/", len(self.files), end = "\r")
                    img = PIL.Image.open(os.path.join(self.folder, file))
                    img = self.transform(img)
                    self.preloaded_imgs.append(img)

                print("PRELOAD DONE")
                self.preloaded_imgs = torch.stack(self.preloaded_imgs)
                torch.save(self.preloaded_imgs, f"preprocessed/preloaded{max_image_count}_{self.img_size}.pt")
                print("SAVED PRELOADED")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if len(self.preloaded_imgs) > 0:
            return self.preloaded_imgs[idx]
        img = PIL.Image.open(os.path.join(self.folder, self.files[idx]))
        img = self.transform(img)
        return img