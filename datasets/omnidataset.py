import io
import json
import os

import h5py
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class OmniDataset(Dataset):

    def __init__(self,  root_dir, subset_name, transform=None, whitebg=True, split="train"):
        super().__init__()
        self.transform = transform
        w = "w" if whitebg else ""
        self.h5 = h5py.File(os.path.join(root_dir, f"data{w}.hdf5"), 'r')
        data_train = pd.read_csv(os.path.join(root_dir, f"dataset{w}_train.csv"))
        data_test = pd.read_csv(os.path.join(root_dir, f"dataset{w}_test.csv"))
        with open(os.path.join(root_dir,"sorted_objs.json"), "r") as outfile:
            obj_view_info = json.load(outfile)
            obj_available = list(obj_view_info.keys())
        if split == "test":
            self.dataset = data_test
        elif split == "train":
            self.dataset = data_train
        else:
            self.dataset = pd.concat([data_train, data_test])

        self.cat_map = {index: k for k, (index, _) in enumerate(data_train.groupby("category").count().iterrows())}
        self.inv_cat_map = {k: index for index, k in self.cat_map.items()}


        self.inv_obj_map = {k: index for k, (index, _) in enumerate(data_train.groupby("object").count().iterrows())}
        l = len(self.inv_obj_map)
        self.inv_obj_map.update({k+l: index for k, (index, _) in enumerate(data_test.groupby("object").count().iterrows())})
        self.obj_map = {index: k for k, index in self.inv_obj_map.items()}

        available = [self.inv_obj_map[int(o)] for o in obj_available]
        self.dataset = self.dataset.query('object in @available').reset_index(drop=True)
        self.dataset = self.dataset.loc[(self.dataset["category"] != "teddy") & (self.dataset["category"] != "teddy_bear")].reset_index(drop=True)
        self.dataset["original_index"] = self.dataset.index
        self.dataset["original_y"] = self.dataset["y"]
        # with open(os.path.join(root_dir, f"omni_canonical_ratio_axis.json"), "r") as outfile:
        #     canonical_data = json.load(outfile)
        # with open(os.path.join(root_dir, f"omni_canonical_ratio_axis_test.json"), "r") as outfile:
        #     canonical_data.update(json.load(outfile))
        # with open(os.path.join(root_dir, f"sorted_objs.json"), "r") as outfile:
        #     obj_info = json.load(outfile)

        mod_y = []
        for i in range(len(self.dataset)):
            canonical_obj_info = obj_view_info[str(self.obj_map[self.dataset.loc[i, "object"]])]
            mod_y.append(0 if not canonical_obj_info[-1] else 90)
        mod_y = np.array(mod_y)
        self.dataset["y"] = (self.dataset["y"].values + 180 + mod_y) % 360 - 180

        if subset_name != "all":
            if subset_name == "front":
                all_angles = [90, -90]
            elif subset_name == "side":
                all_angles = [-180, 0, 180]
            elif subset_name == "quarter":
                all_angles = [-135, -45, 45, 135]
            elif isinstance(subset_name, int):
                all_angles = [subset_name]
            else:
                raise Exception("subset not available")
            selected_views = self.get_specific_views(all_angles)
            self.dataset = self.dataset.iloc[selected_views].reset_index(drop=True)

        print(subset_name, len(self))

    def get_specific_views(self, all_angles):
        all_idxs = []
        for name, group in self.dataset.groupby("object"):
            # positives[name] = {"frames":[], "xs":[]}
            xs = np.asarray(group.get("y"))
            idxs = np.asarray(group.get("original_index"))

            min_ind_a = 10000
            for a in all_angles:
                ind_a= np.abs(xs - np.sign(xs)*a) if a == 180 else np.abs(xs - a)
                if np.min(ind_a) < min_ind_a:
                    indx = idxs[np.argmin(ind_a)]
            all_idxs.append(indx)
        return all_idxs

    def __len__(self):
        return len(self.dataset)


    def get_image(self, category, obj, frame_sample, v):
        return Image.open(io.BytesIO(self.h5.get(category).get(obj)[frame_sample]))

    def __getitem__(self, idx):
        r = self.dataset.iloc[idx]
        category, obj, frame, length= r.loc["category"], r.loc["object"], r.loc["frame"], r.loc["length"]
        img = self.get_image(category, obj, frame, r.loc["original_y"])
        imgt = self.transform(img)
        return imgt, self.cat_map[category], self.obj_map[obj], self.dataset.loc[idx, "y"]
