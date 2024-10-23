import argparse
import os
import random

import PIL
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default="/home/fias/postdoc/datasets/shapebias", type=str, help="dataset root")
parser.add_argument('--name', default="im_obj_feat", choices=["im_obj_feat","obj_text","shape_text"], type=str, help="dataset root")
args = parser.parse_args()




names = args.name.split("_")
type1, type2 = names[0], names[1]
type_to_dir = {"obj": "geirhos-masks","feat": "textures", "shape":"novel-masks", "text": "brodatz-textures"}

dest_path = os.path.join(args.data_root, args.name)
if not os.path.exists(dest_path):
    os.makedirs(dest_path)


if args.name == "im_obj_feat":
    src_path = os.path.join(args.data_root, "images")
    categories = [c for c in os.listdir(src_path)]
    for c in categories:
        for ims1 in os.listdir(os.path.join(src_path, c)):
            for c2 in categories:
                if c2 == c:
                    continue

                ims2 = [im2 for im2 in os.listdir(os.path.join(src_path, c2)) ]
                ims2 = ims2[random.randint(0, len(ims2)-1)]


                im1 = PIL.Image.open(os.path.join(src_path, c, ims1))
                im2 = PIL.Image.open(os.path.join(src_path, c2, ims2))

                shapes = sorted(os.listdir(os.path.join(args.data_root, "geirhos-masks",c )),key=lambda _: random.random())[0]
                texts = sorted(os.listdir(os.path.join(args.data_root, "textures",c2 )),key=lambda _: random.random())[0]

                shape = PIL.Image.open(os.path.join(args.data_root, "geirhos-masks", c, shapes))
                text = PIL.Image.open(os.path.join(args.data_root, "textures", c2, texts))
                shape = torchvision.transforms.functional.to_tensor(shape)
                text = torchvision.transforms.functional.to_tensor(text)
                text = torchvision.transforms.functional.resize(text, 256)
                text = torchvision.transforms.functional.center_crop(text, (224,224))
                shapetext = text * (1-shape)


                triplet_path = os.path.join(dest_path, f'{c}_{c2}_{ims1.split("_")[2]}_{ims2.split("_")[2]}_{shapes.split(".")[0]}_{texts.split("_")[2]}')
                if not os.path.exists(triplet_path):
                    os.makedirs(triplet_path)
                im1.save(os.path.join(triplet_path,"0.png"))
                im2.save(os.path.join(triplet_path,"1.png"))
                torchvision.utils.save_image(shapetext, os.path.join(triplet_path,"2.png"))

if args.name == "obj_text":
    src_path = os.path.join(args.data_root, "geirhos-masks")
    categories = [c for c in os.listdir(src_path)]
    for c in categories:
        for ims1 in os.listdir(os.path.join(src_path, c)):
            for c2 in categories:
                if c2 == c:
                    continue

                ims2 = [im2 for im2 in os.listdir(os.path.join(src_path, c2)) ]
                ims2 = ims2[random.randint(0, len(ims2)-1)]


                im1 = PIL.Image.open(os.path.join(src_path, c, ims1))
                im2 = PIL.Image.open(os.path.join(src_path, c2, ims2))

                texts = sorted(os.listdir(os.path.join(args.data_root, "brodatz-textures")),key=lambda _: random.random())

                text1 = PIL.Image.open(os.path.join(args.data_root, "brodatz-textures", texts[0]))
                text2 = PIL.Image.open(os.path.join(args.data_root, "brodatz-textures", texts[1]))

                shape1 = torchvision.transforms.functional.to_tensor(im1)
                shape2 = torchvision.transforms.functional.to_tensor(im2)

                text1 = torchvision.transforms.functional.to_tensor(text1)[:3]
                text1 = torchvision.transforms.functional.center_crop(text1, (224,224))
                shapetext1 = text1 * (1-shape1)
                shapetext2 = text1 * (1-shape2)

                text2 = torchvision.transforms.functional.to_tensor(text2)[:3]
                text2 = torchvision.transforms.functional.center_crop(text2, (224,224))
                shapetext3 = text2 * (1-shape1)


                triplet_path = os.path.join(dest_path, f'{c}_{c2}_{ims1.split(".")[0]}_{ims2.split(".")[0]}_{texts[0].split(".")[0]}_{texts[1].split(".")[0]}')
                if not os.path.exists(triplet_path):
                    os.makedirs(triplet_path)
                torchvision.utils.save_image(shapetext1, os.path.join(triplet_path,"0.png"))
                torchvision.utils.save_image(shapetext2, os.path.join(triplet_path,"1.png"))
                torchvision.utils.save_image(shapetext3, os.path.join(triplet_path,"2.png"))


if args.name == "shape_text":
    src_path = os.path.join(args.data_root, "novel-masks")
    categories = [c for c in os.listdir(src_path)]
    for ims1 in categories:
        for ims2 in categories:
            if ims1 == ims2:
                continue

            im1 = PIL.Image.open(os.path.join(src_path, ims1))
            im2 = PIL.Image.open(os.path.join(src_path, ims2))

            for texts1, texts2 in zip(os.listdir(os.path.join(args.data_root, "brodatz-textures")),
                                      sorted(os.listdir(os.path.join(args.data_root, "brodatz-textures")),key=lambda _: random.random())):
                if texts1 == texts2:
                    continue
                text1 = PIL.Image.open(os.path.join(args.data_root, "brodatz-textures", texts1))
                text2 = PIL.Image.open(os.path.join(args.data_root, "brodatz-textures", texts2))

                shape1 = torchvision.transforms.functional.to_tensor(im1)[:3]
                shape2 = torchvision.transforms.functional.to_tensor(im2)[:3]

                text1 = torchvision.transforms.functional.to_tensor(text1)[:3]
                text1 = torchvision.transforms.functional.center_crop(text1, (224,224))
                shapetext1 = text1 * (1-shape1)
                shapetext2 = text1 * (1-shape2)

                text2 = torchvision.transforms.functional.to_tensor(text2)[:3]
                text2 = torchvision.transforms.functional.center_crop(text2, (224,224))
                shapetext3 = text2 * (1-shape1)


                triplet_path = os.path.join(dest_path, f'{ims1.split(".")[0]}_{ims2.split(".")[0]}_{texts1.split(".")[0]}_{texts2.split(".")[0]}')
                if not os.path.exists(triplet_path):
                    os.makedirs(triplet_path)
                torchvision.utils.save_image(shapetext1, os.path.join(triplet_path,"0.png"))
                torchvision.utils.save_image(shapetext2, os.path.join(triplet_path,"1.png"))
                torchvision.utils.save_image(shapetext3, os.path.join(triplet_path,"2.png"))