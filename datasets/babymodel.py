import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class IndividualObjectsDataset(Dataset):

    label_to_class = {0: "airplane", 1: "car", 2: "chair", 3: "cup", 4: "dog", 5: "donkey", 6: "duck", 7: "hat"}
    all_subsets = ["blurred", "features", "geons", "realistic", "silhouette"]

    def __init__(self, root_dir, subset_name, transform=None):
        """
        Returns a torch dataset object for the images in a given subset_name.

        Args:
            subset_name (str): Choose "blurred", "boxedfeatures", "geons", "realistic", "silhouette", "all". "all" will return all the images from all the subsets.
            root_dir (str, optional): where subdirectories (subsets) for the data can be found. Defaults to "./individual_objects/".
            transform (torch.transform, optional): torch compose . Defaults to None.
            resize (int, optional): resizes the image while loading the .jpg file. Defaults to 300.
        """
        assert subset_name in self.all_subsets + ["all"], "subset should be 'blurred', 'geons', 'realistic', 'silhouette or 'all"
        self.class_to_label = {n: i for (i, n) in self.label_to_class.items()}

        self.root_dir = root_dir
        self.transform = transform

        self.images, self.labels, self.image_file_names, self.subset, self.name_to_id = [], [], [], [], {}
        if subset_name != "all":
            self.images, self.labels, self.image_file_names = self.return_images_labels(subset_name)
            self.subset = [subset_name] * len(self.images)
        else:
            for subset in self.all_subsets:
                images, labels, image_file_names = self.return_images_labels(subset)
                self.images.extend(images)
                self.labels.extend(labels)
                self.image_file_names.extend(image_file_names)
                self.subset.extend([subset] * len(images))
        print("Dataset of ", subset_name)

    def return_images_labels(self, subset_name):
        self.data_path = os.path.join(self.root_dir, subset_name)
        image_file_names = [f for f in os.listdir(self.data_path)
                                 if os.path.isfile(os.path.join(self.data_path, f))
                                 and os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
        self.name_to_id.update({subset_name+name: i+len(self.images) for i, name in enumerate(image_file_names)})
        images = [Image.open(os.path.join(self.data_path, f)).convert("RGB")
                  for f in image_file_names]
        labels = [self.class_to_label[i]
                  for j in image_file_names
                  for i in self.class_to_label.keys() if i in j.lower()]


        return images, labels, image_file_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, label = self.images[idx], self.labels[idx]
        subset, image_id = self.subset[idx], self.image_file_names[idx]

        label = torch.tensor(label).float()

        if self.transform:
            img = self.transform(img)
        return img, label, self.name_to_id[subset+image_id]
