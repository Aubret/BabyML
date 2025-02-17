import os

import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image

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

class ShepardMetzler(Dataset):

    label_to_class = {i+1 : i for i in range(48)}
    all_subsets = ["features", "rotated", "mirror"]

    def __init__(self, root_dir, subset_name, transform=None, whitebg=True, rotation=[]):

        self.subset_suffix = {"features": "0.jpg", "rotated": "0.jpg", "mirror": "R.jpg"}
        assert subset_name in list(self.all_subsets) + ["all"]
        self.class_to_label = {n: i for (i, n) in self.label_to_class.items()}

        self.root_dir = root_dir
        self.transform = transform
        self.rotation = rotation

        self.labels, self.image_file_names, self.subset, self.name_to_id = [], [], [], {}
        if subset_name != "all":
            self.labels, self.rotations, self.image_file_names = self.return_images_labels(subset_name)
            self.subset = [subset_name] * len(self.labels)
        else:
            for subset in self.all_subsets:
                labels, image_file_names = self.return_images_labels(subset)
                # self.images.extend(images)
                self.labels.extend(labels)
                self.image_file_names.extend(image_file_names)
                self.subset.extend([subset] * len(labels))
        # self.box_features = (75, 0, 400, 426)
        # self.box_features = (75, 0, 400, 426)
        self.box_features = (50, 0, 425, 426)
        # self.box_others = (404, 0, 739, 426)
        self.box_others = (408, 0, 800, 426)

        if whitebg:
            self.transform = torchvision.transforms.Compose([BlackWhite(), transform])
        # print(self.image_file_names)
        # print("Dataset of ", subset_name)

    def return_images_labels(self, subset_name):
        image_file_names = []
        # for l, c in self.label_to_class.items():
        data_path = self.root_dir
        prefix = self.subset_suffix[subset_name]
        files = [f for f in sorted(os.listdir(data_path)) if f.endswith(prefix)]
        if self.rotation:
            # files2 = []
            # for f in files:
            #     for r
            files = [f  for f in files for r in self.rotation if "_"+str(r) in f]
        else:
            files = [f for f in files if "_0" not in f]

        self.name_to_id.update({subset_name+name: i+len(self.name_to_id) for i, name in enumerate(files)})
        labels = [int(f.split("_")[0]) for f in files]
        rotations = [int(f.split("_")[1].split(".")[0]) for f in files]
        image_file_names += files

        return labels, rotations, image_file_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label =  self.labels[idx]
        img = Image.open(os.path.join(self.root_dir, self.image_file_names[idx])).convert("RGB")
        subset, name = self.subset[idx], self.image_file_names[idx]
        img = img.crop(self.box_features if subset == "features" else self.box_others)
        label = torch.tensor(label).float()
        if self.transform:
            img = self.transform(img)
        return img, label, self.name_to_id[subset+name], self.rotations[idx]












