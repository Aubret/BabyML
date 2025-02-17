import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader


from lightning.fabric.strategies import DDPStrategy
import lightning as L

from datasets import DATASETS
from models.registry import list_models, model_registry
from tools import load_model, get_transforms, str2bool


@torch.no_grad()
def shape_bias(args):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.seed_everything(2)
    fabric.launch()

    dataset = args.dataset
    preprocess = get_transforms(args)
    if args.load in list_models():
        model = model_registry[args.load]()
        if isinstance(model, tuple) and len(model) > 1:
            model, preprocess = model
    else:
        model, preprocess = load_model(args, preprocess)

    dataset = DATASETS[dataset](os.path.join(args.data_root, dataset), transform=preprocess, whitebg=args.whitebg)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    dataloader = fabric.setup_dataloaders(dataloader)

    model = fabric.setup(model)
    model.eval()


    shape_decision = 0
    total_decision = 0
    for im1, im2, im3 in dataloader:
        if args.dataset == "img_img_shapetext":
            f1, f2, f3 = model(im1), model(im2), model(im3)
        else:
            f1, f2, f3 = model(im3), model(im2), model(im1)

        f1_f3 = torch.nn.functional.cosine_similarity(f1,f3, dim=1)
        f2_f3 = torch.nn.functional.cosine_similarity(f2,f3, dim=1)
        shape_decision += (f1_f3 > f2_f3).float().sum()
        total_decision += im1.shape[0]

    return (shape_decision/total_decision).item()


def start_shape_bias(args, log_dir):
    print("Eval shape bias", args.dataset, args.load)
    name_test = args.load.split('/')[-1].split(".")[0] if not args.load in list_models() else args.load
    with open(os.path.join(log_dir, f"{name_test}_{args.dataset}_shape_bias.csv"), "w") as f:
        wcsv = csv.writer(f)
        acc = shape_bias(args)
        wcsv.writerow([acc])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--dataset', default="shape_simpletext", type=str)
    parser.add_argument('--whitebg', default=False, type=str2bool)
    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    start_shape_bias(args, log_dir)