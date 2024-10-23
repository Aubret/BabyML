import csv
import os
import random

import torch
from torch.utils.data import DataLoader

import config

from datasets import DATASETS
from lightning.fabric.strategies import DDPStrategy
import lightning as L

from tools import BACKBONES, load_model, add_head, get_features, get_transforms


@torch.no_grad()
def odd_one_out(args):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.seed_everything(args.seed)
    fabric.launch()

    preprocess = get_transforms(args.dataset)
    model, preprocess = load_model(args,  preprocess)
    print(preprocess)

    dataset = DATASETS[args.dataset](args.data_root, subset_name=args.subset, transform=preprocess)
    datasetf = DATASETS[args.dataset](args.data_root, subset_name="realistic", transform=preprocess)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    dataloaderf = DataLoader(datasetf, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    dataloader, dataloaderf = fabric.setup_dataloaders(dataloader, dataloaderf)


    model = fabric.setup(model)
    model.eval()

    features, labels, img_ids = get_features(dataloader, model, fabric)
    featuresf, labelsf, img_idsf = get_features(dataloaderf, model, fabric)
    print(features.shape, featuresf.shape)

    success = 0
    fails = 0
    for l in labels.unique():
        mask_label = (labels == l)
        mask_no_label = ~mask_label

        mask_labelf = labelsf == l
        ids_label = img_idsf[mask_labelf]
        for id in ids_label:
            # mask_no_id = mask_label & (img_ids != id)
            mask_id = img_idsf == id

            positives = features[mask_label]
            negatives = features[mask_no_label]

            # We do 10 random ooo per image
            # p = positives[torch.randint(0, len(positives), (10, ), device=features.device)]
            # n = negatives[torch.randint(0, len(negatives), (10, ), device=features.device)]
            p = positives[torch.arange(0, len(positives) , device=features.device).repeat(len(negatives)).view(-1)]
            n = negatives[torch.arange(0, len(negatives), device=features.device).view(-1,1).repeat((1, len(positives))).view(-1)]
            main_f = featuresf[mask_id].repeat((len(positives)*len(negatives), 1))

            correct = torch.nn.functional.cosine_similarity(main_f, p, dim=1)
            wrong1 = torch.nn.functional.cosine_similarity(main_f, n, dim=1)
            # wrong2 = torch.nn.functional.cosine_similarity(n, p, dim=1)

            # all_success = ( (correct > wrong1) & (correct > wrong2) ).float().sum()
            all_success = (correct > wrong1).float().sum()
            success += all_success
            fails += len(positives)*len(negatives) - all_success
    return (success / (fails + success)).item()


if __name__ == '__main__':
    ### Support only one gpu/cpu
    config.parser.add_argument('--subset', default="full", type=str)
    args = config.parser.parse_args()
    subset_name = args.subset
    args.log_dir = os.path.join(args.log_dir, args.dataset)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    subset_set = DATASETS[args.dataset].all_subsets + ["all"] if subset_name == "full" else [subset_name]
    name_test = args.load.split('/')[-1].split(".")[0]
    with open(os.path.join(args.log_dir, f"{subset_name}_{name_test}_{args.seed}_{args.dataset}_posout_negin.csv"), "w") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(subset_set)
        acc = []
        for subset in subset_set:
            args.subset = subset
            acc.append(odd_one_out(args))
        wcsv.writerow(acc)