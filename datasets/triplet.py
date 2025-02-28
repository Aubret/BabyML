import os

import PIL
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class BlackWhite(object):
    def __call__(self, img):
        img = img.convert("RGB")
        datas = img.getdata()
        new_image_data = []
        th = 5
        for item in datas:
            if item[0] <= th and item[1] <= th and item[2] <= th:
                new_image_data.append((255, 255, 255))
            else:
                new_image_data.append(item)
        img.putdata(new_image_data)
        return img

class TripletDataset(Dataset):

    def __init__(self, root_dir,  transform=None, whitebg=False):

        self.root_dir = root_dir
        self.transform = transform
        # self.whitebg = whitebg
        self.triplets = list(os.listdir(self.root_dir))

        if whitebg:
            self.transform = torchvision.transforms.Compose([BlackWhite(), transform])


    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet_root = os.path.join(self.root_dir, self.triplets[idx])
        images = list(sorted(os.listdir(triplet_root)))
        images = [PIL.Image.open(os.path.join(triplet_root,im)) for im in images]

        if self.transform:
            return self.transform(images[0]), self.transform(images[1]), self.transform(images[2])
        return images[0], images[1], images[2]
