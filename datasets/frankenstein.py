import os

import torch
from torch.utils.data import Dataset
from PIL import Image

class FrankensteinDataset(Dataset):

    label_to_class = {0: "bear", 1: "bunny", 2: "cat", 3: "elephant", 4: "frog", 5: "lizard", 6: "tiger", 7: "turtle", 8:"wolf"}
    all_subsets = ["features", "frankenstein", "fragmented"]

    def __init__(self, root_dir, subset_name, transform=None):
        """
        Returns a torch dataset object for the images in a given subset_name.

        Args:
            subset_name (str): Choose "blurred", "boxedfeatures", "geons", "realistic", "silhouette", "all". "all" will return all the images from all the subsets.
            root_dir (str, optional): where subdirectories (subsets) for the data can be found. Defaults to "./individual_objects/".
            transform (torch.transform, optional): torch compose . Defaults to None.
            resize (int, optional): resizes the image while loading the .jpg file. Defaults to 300.
        """
        self.subset_prefix = {"features": "", "frankenstein": "f", "fragmented": "o"}
        assert subset_name in list(self.subset_prefix.keys()) + ["all"]
        self.class_to_label = {n: i for (i, n) in self.label_to_class.items()}

        self.root_dir = root_dir
        self.transform = transform

        self.labels, self.image_file_names, self.subset, self.name_to_id = [], [], [], {}
        if subset_name != "all":
            self.labels, self.image_file_names = self.return_images_labels(subset_name)
            self.subset = [subset_name] * len(self.labels)
        else:
            for subset in self.subset_prefix.keys():
                labels, image_file_names = self.return_images_labels(subset)
                # self.images.extend(images)
                self.labels.extend(labels)
                self.image_file_names.extend(image_file_names)
                self.subset.extend([subset] * len(labels))
        print("Dataset of ", subset_name)

    def return_images_labels(self, subset_name):
        labels, image_file_names = [], []
        for l, c in self.label_to_class.items():
            self.data_path = os.path.join(self.root_dir, c)
            prefix = self.subset_prefix[subset_name]+c
            files = [f for f in sorted(os.listdir(self.data_path)) if f.startswith(prefix)]
            self.name_to_id.update({subset_name+name: i+len(self.name_to_id) for i, name in enumerate(files)})
            labels += [l] * len(files)
            image_file_names += files


        return labels, image_file_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label =  self.labels[idx]
        img = Image.open(os.path.join(self.root_dir, self.label_to_class[label], self.image_file_names[idx])).convert("RGB")
        subset, name = self.subset[idx], self.image_file_names[idx]

        label = torch.tensor(label).float()

        if self.transform:
            img = self.transform(img)
        return img, label, self.name_to_id[subset+name]












