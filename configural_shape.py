import argparse
import csv
import os

import lightning as L
import torch
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data import DataLoader

from datasets import DATASETS
from models.registry import list_models, model_registry
from tools import load_model, get_transforms, get_features, str2bool


@torch.no_grad()
def configural_shape(args,  subset):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.seed_everything(0)
    fabric.launch()

    dataset = "frankenstein"
    preprocess = get_transforms(args)
    if args.load in list_models():
        model = model_registry[args.load]()
        if isinstance(model, tuple) and len(model) > 1:
            model, preprocess = model
    else:
        model, preprocess = load_model(args, preprocess)

    dataset_train = DATASETS[dataset](args.data_root, subset_name="features", transform=preprocess)
    dataset_test = DATASETS[dataset](args.data_root, subset_name=subset, transform=preprocess)
    dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    dataloader, dataloader_test = fabric.setup_dataloaders(dataloader, dataloader_test)


    model = fabric.setup(model)
    model.eval()

    features, labels, img_ids = get_features(dataloader, model, fabric)
    features_test, labels_test, img_ids_test = get_features(dataloader_test, model, fabric)

    success = 0
    fails = 0

    for l in labels.unique():
        mask_label_t = (labels_test == l)
        mask_label = (labels == l)

        mask_no_label = ~mask_label

        ids_label = img_ids[mask_label]
        for id in ids_label:
            mask_no_id_t = mask_label_t & (img_ids_test != id)
            mask_id = mask_label & (img_ids == id)

            positives = features_test[mask_no_id_t]
            negatives = features[mask_no_label]

            p = positives[torch.arange(0, len(positives) , device=features.device).repeat(len(negatives)).view(-1)]
            n = negatives[torch.arange(0, len(negatives), device=features.device).view(-1,1).repeat((1, len(positives))).view(-1)]
            main_f = features[mask_id].repeat((len(positives)*len(negatives), 1))

            correct = torch.nn.functional.cosine_similarity(main_f, p, dim=1)
            wrong2 = torch.nn.functional.cosine_similarity(n, p, dim=1)

            all_success = (correct > wrong2).float().sum()
            success += all_success
            fails += len(positives)*len(negatives) - all_success
    return (success / (fails + success)).item()



@torch.no_grad()
def configural_shape9(args,  subset):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.seed_everything(0)
    fabric.launch()

    dataset = "frankenstein"
    preprocess = get_transforms(args)
    if args.load in list_models():
        model = model_registry[args.load]()
        if isinstance(model, tuple) and len(model) > 1:
            model, preprocess = model
    else:
        model, preprocess = load_model(args, preprocess)

    dataset_train = DATASETS[dataset](args.data_root, subset_name="features", transform=preprocess)
    dataset_test = DATASETS[dataset](args.data_root, subset_name=subset, transform=preprocess)
    dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    dataloader, dataloader_test = fabric.setup_dataloaders(dataloader, dataloader_test)


    model = fabric.setup(model)
    model.eval()

    features, labels, img_ids = get_features(dataloader, model, fabric)
    features_test, labels_test, img_ids_test = get_features(dataloader_test, model, fabric)

    success = 0
    fails = 0
    all = 0
    lunique = labels.unique()

    for l in labels.unique():
        mask_label_t = (labels_test == l)
        mask_label = (labels == l)

        # mask_no_label = ~mask_label

        ids_label = img_ids[mask_label]

        lunique_neg = lunique[lunique != l]
        masksneg = [labels == ln for ln in lunique_neg ]
        featureneg = [features[masksneg[ln]] for ln in range(len(masksneg))]
        for id in ids_label:
            mask_no_id_t = mask_label_t & (img_ids_test != id)
            mask_id = mask_label & (img_ids == id)

            positives = features_test[mask_no_id_t]
            # negatives = features[mask_no_label]

            main_f = features[mask_id].repeat(100, 1)
            p = positives[torch.randint(0, len(positives), (100,), device=features.device)]

            correct = torch.nn.functional.cosine_similarity(main_f, p, dim=1)
            bcorr = torch.ones((100,), device=features.device, dtype=torch.bool)
            for fneg in featureneg:
                n = fneg[torch.randint(0, len(fneg), (100,), device=features.device)]
                bcorr = bcorr & (correct > torch.nn.functional.cosine_similarity(p, n, dim=1))

            # all_success = ( (correct > wrong1) & (correct > wrong2) ).float().sum()
            all_success = bcorr.float().sum()
            success += all_success
            all += 100
    print("Num comparison", all)
    return (success / all).item()

def start_configural_shape(args, log_dir, subset_name):
    args.dataset = "frankenstein"
    subset_set = DATASETS[args.dataset].all_subsets + ["all"] if subset_name == "full" else [subset_name]
    name_test = args.load.split('/')[-1].split(".")[0] if not args.load in list_models() else args.load
    with open(os.path.join(log_dir, f"{subset_name}_{name_test}_{args.dataset}_config_shape.csv"), "w") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(subset_set)
        acc = []
        for subset in subset_set:
            acc.append(configural_shape(args, subset) if args.class_number == 2 else configural_shape9(args, subset))
        wcsv.writerow(acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="resources/baker/", type=str)
    parser.add_argument('--log_dir', default="logs", type=str)
    parser.add_argument('--load', default="random", type=str)
    parser.add_argument('--model', default="resnet50", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--dense_features', default=False, type=str2bool)

    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--num_devices', default=1, type=int)
    parser.add_argument('--subset', default="full", type=str)
    parser.add_argument('--class_number', default=9, type=int)
    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, "frankenstein9")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    start_configural_shape(args, log_dir, args.subset)