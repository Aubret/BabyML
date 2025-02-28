import argparse
import csv
import os

import lightning as L
import torch
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data import DataLoader

from datasets import DATASETS
from models import list_models
from models.registry import model_registry
from tools import load_model, get_features, get_transforms


@torch.no_grad()
def caricatures(args, subset):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.seed_everything(0)
    fabric.launch()

    preprocess = get_transforms(args)
    if args.load in list_models():
        model = model_registry[args.load]()
        if isinstance(model, tuple) and len(model) > 1:
            model, preprocess = model
    else:
        model, preprocess = load_model(args, preprocess)

    dataset = DATASETS[args.dataset](args.data_root, subset_name=subset, transform=preprocess)
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
    cpt = 0
    labels_success = []
    for l in labels.unique():
        mask_label = (labels == l)
        mask_no_label = ~mask_label

        mask_labelf = labelsf == l
        ids_label = img_idsf[mask_labelf]

        labl_success = 0
        labl_cpt = 0
        for id in ids_label:
            # mask_no_id = mask_label & (img_ids != id)
            mask_id = img_idsf == id

            positives = features[mask_label]
            negatives = features[mask_no_label]

            p = positives[torch.arange(0, len(positives) , device=features.device).repeat(len(negatives)).view(-1)]
            n = negatives[torch.arange(0, len(negatives), device=features.device).view(-1,1).repeat((1, len(positives))).view(-1)]
            main_f = featuresf[mask_id].repeat((len(positives)*len(negatives), 1))

            correct = torch.nn.functional.cosine_similarity(main_f, p, dim=1)
            wrong1 = torch.nn.functional.cosine_similarity(main_f, n, dim=1)

            labl_success += (correct > wrong1).float().sum()
            labl_cpt += correct.shape[0]
        labels_success.append((labl_success/labl_cpt).cpu().item())

        success += labl_success
        cpt += labl_cpt

    return (success / cpt).item()


@torch.no_grad()
def caricatures_hard(args, subset):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.seed_everything(0)
    fabric.launch()

    preprocess = get_transforms(args)
    if args.load in list_models():
        model = model_registry[args.load]()
        if isinstance(model, tuple) and len(model) > 1:
            model, preprocess = model
    else:
        model, preprocess = load_model(args, preprocess)

    dataset = DATASETS[args.dataset](args.data_root, subset_name=subset, transform=preprocess)
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
    cpt = 0

    lunique = labels.unique()
    for l in lunique:
        mask_label = (labels == l)
        mask_labelf = labelsf == l
        ids_label = img_ids[mask_label]


        lunique_neg = lunique[lunique != l]
        masksneg = [labels == ln for ln in lunique_neg ]
        featureneg = [features[masksneg[ln]] for ln in range(len(masksneg))]
        for id in ids_label:
            mask_no_id_t = mask_labelf & (img_idsf != id)
            mask_id = mask_label & (img_ids == id)

            positives = featuresf[mask_no_id_t]
            main_f = features[mask_id].repeat(100, 1)
            p = positives[torch.randint(0, len(positives), (100,), device=features.device)]

            correct = torch.nn.functional.cosine_similarity(main_f, p, dim=1)
            bcorr = torch.ones((100,), device=features.device, dtype=torch.bool)
            for fneg in featureneg:
                n = fneg[torch.randint(0, len(fneg), (100,), device=features.device)]
                bcorr = bcorr & (correct > torch.nn.functional.cosine_similarity(p, n, dim=1))

            all_success = bcorr.float().sum()
            success += all_success
            cpt += 100
    print("Num comparison", cpt)
    return (success / cpt).item()

def start_caricatures(args, log_dir, subset_name):
    subset_set = DATASETS[args.dataset].all_subsets if subset_name == "full" else [subset_name]
    print(subset_name,subset_set)
    name_test = args.load.split('/')[-1].split(".")[0] if not args.load in list_models() else args.load
    with open(os.path.join(log_dir, f"{subset_name}_{name_test}_{args.dataset}_caricatures.csv"), "w") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(subset_set)
        acc = []
        for subset in subset_set:
            acc.append(caricatures(args, subset) if args.difficulty == "simple" else caricatures_hard(args, subset))
        wcsv.writerow(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="resources/BabyVsModel/image_files/v0", type=str)
    parser.add_argument('--log_dir', default="logs", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--num_devices', default=1, type=int)
    parser.add_argument('--subset', default="full", type=str)
    parser.add_argument('--load', default="random", type=str)
    parser.add_argument('--difficulty', default="simple", type=str)
    args = parser.parse_args()
    args.dataset = "babymodel"

    subset_name = args.subset
    args.log_dir = os.path.join(args.log_dir, args.dataset+("hard" if args.difficulty == "hard" else ""))
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    start_caricatures(args, args.log_dir, subset_name)