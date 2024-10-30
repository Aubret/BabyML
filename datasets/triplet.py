import os

import PIL
import torch
from torch.utils.data import Dataset
from PIL import Image

class TripletDataset(Dataset):

    def __init__(self, root_dir,  transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.triplets = list(os.listdir(self.root_dir))


    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet_root = os.path.join(self.root_dir, self.triplets[idx])
        images = list(sorted(os.listdir(triplet_root)))
        images = [PIL.Image.open(os.path.join(triplet_root,im)) for im in images]

        if self.transform:
            return self.transform(images[0]), self.transform(images[1]), self.transform(images[2])
        return images[0], images[1], images[2]
