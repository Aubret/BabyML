import argparse
import csv
import os
import re
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/..")

import numpy as np
import torchvision.utils

from models import list_models
from models.registry import model_registry


import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from lightning.fabric.strategies import DDPStrategy
import lightning as L

from datasets import DATASETS
from models.heads import ActionMultiLayerProj, MultiLayerProj, MultiLayerProjShortcut
from tools import BACKBONES, load_model, get_transforms, add_head, get_features, hook_dense_features
from torchvision.transforms import v2 as trv2, InterpolationMode

# def load_model(load):
#     home = os.environ['HOME']
#     dest_path = f"{home}/.cache/torch/hub/mental_rotations/"
#     if not os.path.exists(dest_path):
#         os.makedirs(dest_path)
#
#     path_load = "goethe:/scratch/autolearn/aubret/results/imgnet/"
#     dest_path = os.path.join(dest_path, load)
#
#     if os.path.exists(dest_path):
#         model.load_state_dict(torch.load(f"{home}/.cache/torch/hub/checkpoints/byol_r50.pth.tar"))
#         return model

def clean_checkpoint(checkpoint):
    new_state_dict = {}
    for k, w in checkpoint.items():

        if re.search("^model.*", k):
            k = ".".join(k.split(".")[1:])

        if re.search("^sup_lin*", k):
            continue
        # if "action_projector" in k and not "action_projector" in args.keep_proj:
        #     continue
        if "action_head" in k and not "action_head" in args.keep_proj:
            continue

        if re.search("^projector.*", k) and "projector" in args.keep_proj:
            new_k = ".".join(["projector.layers"] + k.split(".")[2:])
            new_state_dict[new_k] = w
        elif "action_projector" in k:
            if "action_projector" in args.keep_proj:
                new_k = ".".join(["head_action.layers"] + k.split(".")[2:])
                new_state_dict[new_k] = w
        elif "equivariant_projector" in k:
            if "equivariant_projector" in args.keep_proj:
                new_k = ".".join(["head_equivariant.layers"] + k.split(".")[2:])
                new_state_dict[new_k] = w
        elif "equivariant_predictor" in k:
            if "equivariant_predictor" in args.keep_proj:
                new_k = ".".join(["head_prediction.layers"] + k.split(".")[2:])
                new_state_dict[new_k] = w
        elif "action_rep_projector" in k:
            new_k = k.replace("net","layers")
            new_state_dict[new_k] = w
        else:
            new_state_dict[k] = w
    return new_state_dict

def complete_head(model):
    add_head(model, args.head)
    # n_out = 2048
    n_out = 512 if args.model == "resnet18" else 2048
    n_features=128
    print(n_out)

    model.projector = MultiLayerProj(2, n_out, 2048, n_features, bias=False)
    if args.load != "random":
        checkpoint = torch.load(args.load, map_location="cpu")
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        checkpoint = clean_checkpoint(checkpoint)


        to_remove = []
        for k in checkpoint.keys():
            if k.startswith("fc."):
                to_remove.append(k)
            if k.startswith("classifier."):
                to_remove.append(k)

        for k in to_remove:
            del checkpoint[k]
        model.load_state_dict(checkpoint, strict=args.load_strict)
    return model



@torch.no_grad()
def invariance(args):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.launch()

    # mean, std, image_size = (0, 0, 0), (1, 1, 1), 224
    mean, std, image_size =  (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
    preprocess = trv2.Compose([trv2.Resize(image_size, interpolation=InterpolationMode.BICUBIC), trv2.CenterCrop(image_size),
                         trv2.ToImage(), trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])


    if args.load in list_models():
        model = model_registry[args.load]()
        if isinstance(model, tuple) and len(model) > 1:
            model, preprocess = model
    else:
        model = BACKBONES[args.model]()
        model = complete_head(model)

    whitebg=False
    dataset = DATASETS[args.dataset](args.data_root, subset_name="features", transform=preprocess, whitebg=whitebg, rotation=args.rotation)
    dataset_pos = DATASETS[args.dataset](args.data_root, subset_name=args.pos_subset, transform=preprocess, whitebg=whitebg, rotation=args.rotation)
    dataset_neg = DATASETS[args.dataset](args.data_root, subset_name=args.neg_subset, transform=preprocess, whitebg=whitebg, rotation=args.rotation)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_pos = DataLoader(dataset_pos, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_neg = DataLoader(dataset_neg, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)

    dataloader, dataloader_pos, dataloader_neg = fabric.setup_dataloaders(dataloader, dataloader_pos, dataloader_neg)

    model = fabric.setup(model)
    model.eval()

    # for img, label, img_id in dataloader:
    #     for i, idi in zip(img, img_id):
    #         torchvision.utils.save_image(i, "/home/fias/postdoc/gym_results/test_images/sheperdblack/"+str(idi.item())+".png")
    #     break
    # for img, label, img_id in dataloader_pos:
    #     for i, idi in zip(img, img_id):
    #         torchvision.utils.save_image(i, "/home/fias/postdoc/gym_results/test_images/sheperdblack2/"+str(idi.item())+".png")
    #     break
    # for img, label, img_id in dataloader_neg:
    #     for i, idi in zip(img, img_id):
    #         torchvision.utils.save_image(i, "/home/fias/postdoc/gym_results/test_images/sheperdblack3/"+str(idi.item())+".png")
    #     break
    # return

    features, labels, img_ids = get_features(dataloader, model, fabric)
    features_pos, labels_pos, img_ids_pos = get_features(dataloader_pos, model, fabric)
    features_neg, labels_neg, img_ids_neg = get_features(dataloader_neg, model, fabric)

    # assert to verify the shuffle=False works well
    assert (img_ids_neg == img_ids_pos).float().sum() == img_ids_pos.shape[0]
    assert (img_ids == img_ids_pos).float().sum() == img_ids_pos.shape[0]

    assert (labels == labels_pos).float().sum() == labels.shape[0]


    correct = torch.nn.functional.cosine_similarity(features, features_pos, dim=1)
    wrong1 = torch.nn.functional.cosine_similarity(features, features_neg, dim=1)
    # wrong2 = torch.nn.functional.cosine_similarity(features_pos, features_neg, dim=1)
    # naive_acc= ((correct >= wrong1) & (correct >= wrong2)).float().mean().item()
    naive_acc= (correct > wrong1).float().sum().item()
    # print(features.shape[0], naive_acc/features.shape[0])
    # print( (correct > wrong1).float().mean().item())
    if not hasattr(model, "projector"):
        return np.array([naive_acc])/features.shape[0]
    res = [naive_acc] + [0 for _ in range(len(model.projector.layers))]
    for f, f_p, f_n in zip(features.split(64), features_pos.split(64), features_neg.split(64)):
        for i, layer in enumerate(model.projector.layers):
            f_proj, f_p_proj, f_n_proj = layer(f), layer(f_p), layer(f_n)
            res[1+i] += (torch.nn.functional.cosine_similarity(f_proj, f_p_proj, dim=1) >
                      torch.nn.functional.cosine_similarity(f_proj, f_n_proj, dim=1)).float().sum().item()

    return np.array(res)/features.shape[0]



@torch.no_grad()
def invariance_v2(args):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.launch()

    # mean, std, image_size = (0, 0, 0), (1, 1, 1), 224
    mean, std, image_size =  (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
    preprocess = trv2.Compose([trv2.Resize(image_size, interpolation=InterpolationMode.BICUBIC), trv2.CenterCrop(image_size),
                         trv2.ToImage(), trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])


    if args.load in list_models():
        model = model_registry[args.load]()
        if isinstance(model, tuple) and len(model) > 1:
            model, preprocess = model
    else:
        model = BACKBONES[args.model]()
        model = complete_head(model)

    if args.dense_features:
        model = hook_dense_features(model)
    # if args.dense_features:
    #     if has
    #     model.

    whitebg=False
    dataset = DATASETS[args.dataset](args.data_root, subset_name=args.pos_subset, transform=preprocess, whitebg=whitebg, rotation=["0"])
    dataset_pos = DATASETS[args.dataset](args.data_root, subset_name=args.pos_subset, transform=preprocess, whitebg=whitebg, rotation=args.rotation)
    dataset_neg = DATASETS[args.dataset](args.data_root, subset_name=args.neg_subset, transform=preprocess, whitebg=whitebg, rotation=args.rotation)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_pos = DataLoader(dataset_pos, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_neg = DataLoader(dataset_neg, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)

    dataloader, dataloader_pos, dataloader_neg = fabric.setup_dataloaders(dataloader, dataloader_pos, dataloader_neg)

    model = fabric.setup(model)
    model.eval()

    imgs, labels, img_ids = [], [], []
    imgsp, labelsp, img_idsp = [], [], []
    imgsn, labelsn, img_idsn = [], [], []

    for r in tqdm(dataloader):
        imgs.append(r[0])
        labels.append(r[1])
        img_ids.append(r[2])
    for rp in tqdm(dataloader_pos):
        imgsp.append(rp[0])
        labelsp.append(rp[1])
        img_idsp.append(rp[2])

    for rn in tqdm(dataloader_neg):
        imgsn.append(rn[0])
        labelsn.append(rn[1])
        img_idsn.append(rn[2])

    imgs = torch.cat(imgs)
    imgsp = torch.cat(imgsp)
    imgsn = torch.cat(imgsn)

    labels = torch.cat(labels)
    labelsp = torch.cat(labelsp)
    labelsn = torch.cat(labelsn)

    features, featuresp, featuresn = [], [], []
    success, cpt = 0, 0
    for id in range(1, 48, 1):
        mask = labels == id
        maskp = labelsp == id
        maskn = labelsn == id



        inputsp = imgsp[maskp]
        featuresp = model(inputsp)
        if args.dense_features:
            featuresp = model.dense_features.flatten(1)

        inputsn = imgsn[maskn]
        featuresn =model(inputsn)
        if args.dense_features:
            featuresn = model.dense_features.flatten(1)

        inputs = imgs[mask]
        features= model(inputs).repeat(featuresn.shape[0],1)
        if args.dense_features:
            features = model.dense_features.flatten(1)

        correct = torch.nn.functional.cosine_similarity(features, featuresp, dim=1)
        wrong = torch.nn.functional.cosine_similarity(features, featuresn, dim=1)
        success += (correct > wrong).float().sum().item()
        cpt += featuresp.shape[0]

    return np.array([success])/cpt


def start_invariance(args, log_dir):
    if not args.load in list_models():
        splits = args.load.split('/')
        name_test = splits[-4] + "_" + splits[-1].split(".")[0]
    else:
        name_test = args.load


    with open(os.path.join(log_dir, f"{args.pos_subset}_{name_test}_{args.dataset}_ooo_inv{args.rotation if args.rotation else ''}.csv"), "w") as f:
        wcsv = csv.writer(f)
        acc = invariance_v2(args)
        wcsv.writerow(acc)


if __name__ == '__main__':
    def str2table(v):
        return v.split(',')


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

    # General
    parser.add_argument('--data_root', default="../datasets/ShepardMetzler/", type=str)
    parser.add_argument('--log_dir', default="logs", type=str)
    parser.add_argument('--load', default="random", type=str)
    parser.add_argument('--model', default="resnet50", type=str)
    parser.add_argument('--head', default="", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--load_strict', default=True, type=str2bool)
    parser.add_argument('--rotation', default=[], type=str2table)
    parser.add_argument('--dense_features', default=False, type=str2bool)
    parser.add_argument('--keep_proj', default=["projector"], type=str2table)

    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--num_devices', default=1, type=int)

    args = parser.parse_args()
    args.dataset = "shepardmetzler"
    args.pos_subset = "rotated"
    args.neg_subset = "mirror"
    # assert args.head == "action_prediction", "Need action prediction module"
    args.log_dir = os.path.join(args.log_dir, args.dataset, "inv")
    if args.dense_features:
        args.log_dir += "dense"
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    start_invariance(args, args.log_dir)

