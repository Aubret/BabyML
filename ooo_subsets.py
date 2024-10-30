import csv
import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import config
from lightning.fabric.strategies import DDPStrategy
import lightning as L

from datasets import DATASETS
from tools import BACKBONES, load_model, get_transforms, add_head, get_features


@torch.no_grad()
def ooo_subsets(args):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.launch()

    preprocess = get_transforms(args.dataset)
    model, preprocess = load_model(args, preprocess)

    dataset = DATASETS[args.dataset](args.data_root, subset_name="features", transform=preprocess)
    dataset_pos = DATASETS[args.dataset](args.data_root, subset_name=args.pos_subset, transform=preprocess)
    dataset_neg = DATASETS[args.dataset](args.data_root, subset_name=args.neg_subset, transform=preprocess)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_pos = DataLoader(dataset_pos, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_neg = DataLoader(dataset_neg, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)

    dataloader, dataloader_pos, dataloader_neg = fabric.setup_dataloaders(dataloader, dataloader_pos, dataloader_neg)


    model = load_model(args)
    model = fabric.setup(model)
    model.eval()

    features, labels, img_ids = get_features(dataloader, model, fabric)
    features_pos, labels_pos, img_ids_pos = get_features(dataloader_pos, model, fabric)
    features_neg, labels_neg, img_ids_neg = get_features(dataloader_neg, model, fabric)

    # assert to verify the shuffle=False works well
    assert (img_ids_neg == img_ids_pos).float().sum() == img_ids_pos.shape[0]
    assert (img_ids == img_ids_pos).float().sum() == img_ids_pos.shape[0]


    correct = torch.nn.functional.cosine_similarity(features, features_pos, dim=1)
    wrong1 = torch.nn.functional.cosine_similarity(features, features_neg, dim=1)
    wrong2 = torch.nn.functional.cosine_similarity(features_pos, features_neg, dim=1)
    return ((correct >= wrong1) & (correct >= wrong2)).float().mean().item()

if __name__ == '__main__':
    config.parser.add_argument('--pos_subset', default="rotated", type=str)
    config.parser.add_argument('--neg_subset', default="mirror", type=str)
    args = config.parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, args.dataset)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


    name_test = args.load.split('/')[-1].split(".")[0]
    with open(os.path.join(args.log_dir, f"{args.pos_subset}_{name_test}_{args.dataset}_ooo_subsets.csv"), "w") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["mean ooo"])
        acc = [ooo_subsets(args)]
        wcsv.writerow(acc)