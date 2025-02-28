import argparse
import csv
import math
import os
import random
import sys

import numpy as np
import torchvision

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/..")


import torch
from torch.utils.data import DataLoader

import config
from lightning.fabric.strategies import DDPStrategy
import lightning as L

from datasets import DATASETS
from tools import BACKBONES, load_model, get_transforms, add_head, get_features
from torchvision import transforms

parser = argparse.ArgumentParser()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise Exception('Boolean value expected.')

# General
parser.add_argument('--data_root', default="data", type=str)
parser.add_argument('--log_dir', default="logs", type=str)
parser.add_argument('--load', default="random", type=str)
parser.add_argument('--model', default="resnet50", type=str)
parser.add_argument('--head', default="", type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--load_strict', default=True, type=str2bool)

parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--num_devices', default=1, type=int)


def get_action(src1, src2, img):
    cy1 = src1[0] #+ 0.5 * src1[2]
    cx1 = src1[1] #+ 0.5 * src1[3]
    h1, w1 = src1[2], src1[3]

    cy2 = src2[0] #+ 0.5 * src2[2]
    cx2 = src2[1] #+ 0.5 * src2[3]
    h2, w2 = src2[2], src2[3]


    s = img.shape[2], img.shape[3]
    d_cx = (cx1 - cx2)/s[1]
    d_cy = (cy1 - cy2)/s[0]
    d_sx = math.sqrt(w1 / w2)
    d_sy = math.sqrt(h1 / h2)
    return torch.tensor((d_cx, d_cy, d_sx, d_sy, 0))


@torch.no_grad()
def action(args):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.seed_everything(args.seed)
    fabric.launch()

    preprocess = get_transforms(args.dataset)
    model, preprocess = load_model(args, preprocess)
    # image_size = 224
    # t = trv2.Compose([trv2.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC), trv2.ToImage(),
    #                          trv2.ToDtype(torch.float32, scale=True)])
    dataset_test = DATASETS[args.dataset](args.data_root, subset_name=args.subset, transform=preprocess)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, pin_memory=True)
    dataloader_test = fabric.setup_dataloaders(dataloader_test)

    model = fabric.setup(model)
    model.eval()

    a_all = 0
    cpt = 0
    a_loss = 0
    # mean_a = torch.tensor([-0.0836, -0.2031,  1.0251,  1.0222], device=fabric.device)
    mean_a = torch.tensor([-7.1625e-04, -8.0850e-04,  1.0251e+00,  1.0222e+00,  5.1843e-01], device=fabric.device)
    # var_a = torch.tensor([9.6652e+03, 5.4049e+03, 5.1830e-02, 4.6630e-02], device=fabric.device)
    var_a = torch.tensor([0.0344, 0.0243, 0.0518, 0.0466, 0.2485], device=fabric.device)
    for img, label, img_id in dataloader_test:
        p1s = (0, 0, 112, 112), (112, 0, 112, 112), (0,0,112,112)
        p2s = (112, 112, 112, 112), (0, 112, 112, 112), (0,112,112,112)
        for p1, p2, in zip(p1s, p2s):
            img1 = transforms.functional.crop(img, *p1)
            img1 = transforms.functional.resize(img1, 224, interpolation=transforms.functional.InterpolationMode.BICUBIC)

            img2 = transforms.functional.crop(img, *p2)
            img2 = transforms.functional.resize(img2, 224, interpolation=transforms.functional.InterpolationMode.BICUBIC )
            # print(torch.max(img), torch.max(img1), torch.max(img2))
            # torchvision.utils.save_image(img1[0], "/home/fias/postdoc/gym_results/test_images/babymodel/test.png")

            # plt.imshow(np.transpose(img[0].cpu().numpy(), (1,2,0)))
            # plt.show()
            # plt.imshow(np.transpose(img1[0].cpu().numpy(), (1,2,0)))
            # plt.show()
            # plt.imshow(np.transpose(img2[0].cpu().numpy(), (1,2,0)))
            # plt.show()


            f1 = model(img1)
            f2 = model(img2)

            pred_a = model.head.forward_all(f1, f2)
            # print("---")
            # print(p1, p2)

            true_a = (get_action(p1, p2, img).to(fabric.device) - mean_a)/torch.sqrt(var_a)
            true_a = true_a.unsqueeze(0).repeat((len(pred_a), 1))
            a_loss += torch.nn.functional.mse_loss(pred_a, true_a)
            a_all += pred_a.mean(dim=0)
            cpt += 1
    print(a_all/cpt)
    return (a_loss/cpt).item()




if __name__ == '__main__':
    ### Support only one gpu/cpu
    config.parser.add_argument('--subset', default="full", type=str)
    args = config.parser.parse_args()
    subset_name = args.subset
    args.log_dir = os.path.join(args.log_dir, args.dataset, "action")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    subset_set = DATASETS[args.dataset].all_subsets if subset_name == "full" else [subset_name]
    name_test = args.load.split('/')[-1].split(".")[0]
    with open(os.path.join(args.log_dir, f"{subset_name}_{name_test}_{args.seed}_{args.dataset}_ooo{args.head}_action.csv"), "w") as f:
        wcsv = csv.writer(f)

        srow = []
        for s in subset_set:
            # srow.append(f"a_{s}")
            srow.append(f"aloss_{s}")
        wcsv.writerow(srow)
        acc = []
        for subset in subset_set:
            args.subset = subset
            loss = action(args)
            # acc.append(a)
            acc.append(loss)
        wcsv.writerow(acc)