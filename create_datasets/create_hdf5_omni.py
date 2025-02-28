import argparse
import copy
import csv
import io
import json
import tempfile
import time

import cv2
import h5py
import numpy as np
import scipy
import os

from PIL import Image

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise Exception('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="resources/OmniDataset")
parser.add_argument("--build_hdf5", type=str2bool, default=True)
parser.add_argument("--start_from", type=int, default=0)
parser.add_argument("--stop_at", type=int, default=500)
args=parser.parse_args()

os.makedirs(args.path, exist_ok=True)

train_csv_file = open(os.path.join(args.path,f"datasetw_train.csv"), "w" if args.start_from == 0 else "a")
test_csv_file = open(os.path.join(args.path,f"datasetw_test.csv"), "w" if args.start_from == 0 else "a")
#
csv_writer_train = csv.writer(train_csv_file)
csv_writer_test = csv.writer(test_csv_file)

if args.start_from == 0:
    csv_writer_train.writerow(["category", "object", "view", "frame", "length", "y", "z", "x","background"])
    csv_writer_test.writerow(["category", "object", "view", "frame", "length", "y", "z", "x","background"])


def transform_angles(f):
    tf = np.array(f["transform_matrix"])
    tf = tf[0:3]
    tf = tf[:, 0:3]
    tf[0] = tf[0] / np.linalg.norm(tf[0])
    tf[1] = tf[1] / np.linalg.norm(tf[1])
    tf[2] = tf[2] / np.linalg.norm(tf[2])
    # rot_gen = scipy.spatial.transform.Rotation.from_matrix(tf)
    rot_gen = scipy.spatial.transform.Rotation.from_matrix(tf)
    return rot_gen.as_euler("yzx", degrees=True)


if args.build_hdf5:
    dataset = h5py.File(os.path.join(args.path, f"dataw.hdf5"), "a")

src_path = 'resources/OpenXDLab___OmniObject3D-New/raw/blender_renders_24_views/img'

for k, category in enumerate(sorted(os.listdir(src_path))):
    if k < args.start_from or k >= args.stop_at:
        continue
    if k% 2 == 0:
        # I had cache issues in my case
        os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')
    category_path = os.path.join(src_path, category)
    if not os.path.isdir(category_path):
        continue
    num_obj = len(next(os.walk(category_path))[1])
    if args.build_hdf5:
        category_dataset = dataset.create_group(category)

    # store_np = []
    all_f = 0
    for i, object_name in enumerate(sorted(os.listdir(category_path))):


        object_path = os.path.join(category_path, object_name)
        json_file = json.load(open(os.path.join(object_path, "transforms.json"), "r"))["frames"]
        num_views = len(json_file)

        store_np = []
        cpt_frames = 0
        test_obj = (float(i) / float(num_obj) > 0.66) or i+1 >= num_obj
        # object_dataset = category_dataset.create_dataset(object_name, shape=(num_views, 224, 224, 3),dtype=np.uint8)
        for f in json_file:
            filename = f["file_path"]
            v_path = os.path.join(object_path, filename)

            if args.build_hdf5:
                im2 = Image.open(v_path).resize((224, 224))
                im = Image.new('RGBA', (224,224), "white")
                im.paste(im2, (0, 5), im2)
                im = im.convert("RGB")
                binary_data = io.BytesIO()
                im.save(binary_data, format="JPEG")
                store_np.append(np.asarray(binary_data.getvalue()))
                im.close()

            yzx = transform_angles(f)
            row = [category, object_name, filename, cpt_frames, num_views, yzx[0], yzx[1], yzx[2]]
            writer = csv_writer_train if not test_obj else csv_writer_test
            writer.writerow(row)
            cpt_frames += 1
            all_f += 1

        if args.build_hdf5:
            object_dataset = category_dataset.create_dataset(object_name, data=np.stack(store_np))

    print(f"end {category}")
train_csv_file.close()
test_csv_file.close()
