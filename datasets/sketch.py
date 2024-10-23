import os

import torch
from torch.utils.data import Dataset
from PIL import Image

class Sketch(Dataset):

    label_to_class = {i+1 : i for i in range(48)}
    all_subsets = ["features","features_test"]

    def __init__(self, root_dir, subset_name, transform=None):
        """
        Returns a torch dataset object for the images in a given subset_name.

        Args:
            subset_name (str): Choose "blurred", "boxedfeatures", "geons", "realistic", "silhouette", "all". "all" will return all the images from all the subsets.
            root_dir (str, optional): where subdirectories (subsets) for the data can be found. Defaults to "./individual_objects/".
            transform (torch.transform, optional): torch compose . Defaults to None.
            resize (int, optional): resizes the image while loading the .jpg file. Defaults to 300.
        """
        assert subset_name in list(self.all_subsets) + ["all"]
        self.class_to_label = {n: i for (i, n) in self.label_to_class.items()}

        self.root_dir = os.path.join(root_dir,"png")
        self.transform = transform

        self.labels, self.image_file_names = self.return_images_labels(subset_name)
        # print(self.image_file_names)
        # print("Dataset of ", subset_name)

    def return_images_labels(self, subset_name):
        labels, image_file_names = [], []
        # for l, c in self.label_to_class.items():
        data_path = self.root_dir

        for d in os.listdir(self.root_dir):
            listfiles = sorted(os.listdir(os.path.join(self.root_dir, d)))
            n_files = len(listfiles)
            for i, img in enumerate(listfiles):
                if subset_name == "features" and i < int(n_files * 0.66):
                    image_file_names.append(img)

                if subset_name == "features_test" and i >= int(n_files * 0.66):
                    image_file_names.append(img)

        self.name_to_id.update({subset_name+name: i+len(self.name_to_id) for i, name in enumerate(files)})
        labels = [int(f.split("_")[0]) for f in files]
        image_file_names += files


        return labels, image_file_names

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
        return img, label, self.name_to_id[subset+name]












