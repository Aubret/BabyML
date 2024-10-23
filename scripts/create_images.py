import argparse
import csv
import os

import PIL.Image
import torchvision.transforms.functional
from torchvision.transforms import InterpolationMode

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default="/home/fias/postdoc/datasets/imgnet/val", type=str, help="ImgNet data root")
args = parser.parse_args()

# textures_wanted = ["plane", "brown_bear", "mountain_bike", "hummingbird", "speedboat",
#                    "water_bottle",  "passenger_car", "tiger_cat", "rocking_chair","wall_clock", "French_bulldog"]
textures_wanted = {"warplane": "airplane", "brown_bear":"bear",
           "mountain_bike":"bicycle", "hummingbird":"bird",
            "speedboat":"boat", "water_bottle":"bottle",
            "passenger_car":"car", "tiger_cat":"cat",
            "rocking_chair":"chair","wall_clock":"clock",
            "French_bulldog":"dog", "African_elephant":"elephant",
           "computer_keyboard":"keyboard", "letter_opener":"knife",
           "Dutch_oven":"oven", "trailer_truck":"truck"}
dict = {}
with open("resources/imgnet_mapping.txt", "r") as f:
    reader = csv.reader(f, delimiter=" ")
    for r in reader:
        dict[r[2]] = r[0]


for t in textures_wanted.keys():
    d = os.path.join(args.data_root,dict[t])
    tmap = textures_wanted[t]

    if not os.path.exists(f"resources/images/{tmap}"):
        os.makedirs(f"resources/images/{tmap}")
    for i, image in enumerate(reversed(os.listdir(d))):
        if not os.path.exists(f"resources/images/{tmap}/{image}"):
            try:
                img = PIL.Image.open(os.path.join(d, image))
            except:
                continue
            img = torchvision.transforms.functional.resize(img, 256, interpolation=InterpolationMode.BICUBIC)
            img = torchvision.transforms.functional.center_crop(img, (224,224))
            img.save(f"resources/images/{tmap}/{image}")
        if i >= 8:
            break






