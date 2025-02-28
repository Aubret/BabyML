import argparse
import csv
import os
import re
import sys


sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/..")
import numpy as np
from tqdm import tqdm
from models import list_models
from models.registry import model_registry
from scripts.metrics import centered_kernel_alignment, compare_batches
import torch
from torch.utils.data import DataLoader
from lightning.fabric.strategies import DDPStrategy
import lightning as L
from datasets import DATASETS
from models.heads import ActionMultiLayerProj, MultiLayerProj, MultiLayerProjShortcut
from tools import BACKBONES, load_model, get_transforms, add_head, get_features, hook_dense_features
from torchvision.transforms import v2 as trv2, InterpolationMode

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
def invariance_max(args):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.launch()

    # mean, std, image_size = (0, 0, 0), (1, 1, 1), 224
    mean, std, image_size =  (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
    preprocess = trv2.Compose([trv2.Resize(image_size, interpolation=InterpolationMode.BICUBIC), trv2.CenterCrop(image_size),
    # preprocess = trv2.Compose([trv2.Resize(image_size), trv2.CenterCrop(image_size),
                         trv2.ToImage(), trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])


    if args.load in list_models():
        # model = model_registry[args.load](n_frame=n_frame)
        model = model_registry[args.load]()
        if isinstance(model, tuple) and len(model) > 1:
            model, preprocess = model
    else:
        model = BACKBONES[args.model]()
        model = complete_head(model)
    if args.dense_features:
        model = hook_dense_features(model)

    whitebg=False
    dataset = DATASETS[args.dataset](args.data_root, subset_name=args.pos_subset, transform=preprocess, whitebg=whitebg, rotation=["0","50","100"])
    dataset_pos = DATASETS[args.dataset](args.data_root, subset_name=args.pos_subset, transform=preprocess, whitebg=whitebg, rotation=["150"])
    dataset_neg = DATASETS[args.dataset](args.data_root, subset_name=args.neg_subset, transform=preprocess, whitebg=whitebg, rotation=["150"])

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_pos = DataLoader(dataset_pos, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_neg = DataLoader(dataset_neg, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)

    dataloader, dataloader_pos, dataloader_neg = fabric.setup_dataloaders(dataloader, dataloader_pos, dataloader_neg)

    model = fabric.setup(model)
    model.eval()


    imgs, labels, img_ids, rotation = [], [], [], []
    imgsp, labelsp, img_idsp,rotationp = [], [], [], []
    imgsn, labelsn, img_idsn, rotationn = [], [], [], []

    for r in tqdm(dataloader):
        imgs.append(r[0])
        labels.append(r[1])
        img_ids.append(r[2])
        rotation.append(r[3])
    for rp in tqdm(dataloader_pos):
        imgsp.append(rp[0])
        labelsp.append(rp[1])
        img_idsp.append(rp[2])
        rotationp.append(rp[3])

    for rn in tqdm(dataloader_neg):
        imgsn.append(rn[0])
        labelsn.append(rn[1])
        img_idsn.append(rn[2])
        rotationn.append(rn[3])

    imgs = torch.cat(imgs)
    imgsp = torch.cat(imgsp)
    imgsn = torch.cat(imgsn)

    labels = torch.cat(labels)
    labelsp = torch.cat(labelsp)
    labelsn = torch.cat(labelsn)

    rotation = torch.cat(rotation)
    rotationp = torch.cat(rotationp)
    rotationn = torch.cat(rotationn)

    features, featuresp, featuresn = [], [], []
    features1, features2, features3 = [], [], []
    for id in range(1, 49, 1):
        mask = labels == id
        maskp = labelsp == id
        maskn = labelsn == id


        inputsm = imgs[mask]
        inputsm = inputsm[torch.argsort(rotation[mask])]

        fp = model(imgsp[maskp])
        if args.dense_features:
            fp = model.dense_features.flatten(1)
        featuresp.append(fp)

        fn = model(imgsn[maskn])
        if args.dense_features:
            fn = model.dense_features.flatten(1)
        featuresn.append(fn)

        f1= model(inputsm[0:1])
        if args.dense_features:
            f1 = model.dense_features.flatten(1)
        features1.append(f1)

        f2 = model(inputsm[1:2])
        if args.dense_features:
            f2 = model.dense_features.flatten(1)
        features2.append(f2)

        f3 = model(inputsm[2:3])
        if args.dense_features:
            f3 = model.dense_features.flatten(1)
        features3.append(f3)

        # featuresp.append(model(imgsp[maskp]))
        # featuresn.append(model(imgsn[maskn]))
        #
        # features1.append(model(inputsm[0:1]))
        # features2.append(model(inputsm[1:2]))
        # features3.append(model(inputsm[2:3]))


    featuresp = torch.cat(featuresp)
    featuresn = torch.cat(featuresn)

    acc = []
    features1, features2, features3 = torch.cat(features1), torch.cat(features2), torch.cat(features3)
    features_all = torch.stack((features1, features2, features3), dim=1)
    correct_all = torch.nn.functional.cosine_similarity(featuresp.unsqueeze(1), features_all, dim=2)
    wrong_all = torch.nn.functional.cosine_similarity(featuresn.unsqueeze(1), features_all, dim=2)

    # print(features_all.shape)
    # correct_all = compare_batches(featuresp, features_all[:,2])
    # wrong_all = compare_batches(featuresn, features_all[:,2])

    # correct_all = -torch.norm(featuresp.unsqueeze(1) - features_all, dim=2)
    # wrong_all = -torch.norm(featuresn.unsqueeze(1) - features_all, dim=2)
    corrects = torch.max(correct_all, dim=1).values
    wrongs = torch.max(wrong_all, dim=1).values
    acc.append((corrects > wrongs).float().sum().item())

    for f in (features1, features2, features3):
        c = torch.nn.functional.cosine_similarity(featuresp, f, dim=1)
        w= torch.nn.functional.cosine_similarity(featuresn, f , dim=1)
        acc.append((c > w).float().sum().item())

    # print(correct_all.squeeze())
    # print(wrong_all.squeeze())
    # acc.append((correct_all < wrong_all).float().sum().item())



    #
    return np.array(acc)/features_all.shape[0]


def start_invariance_max(args, log_dir):
    if not args.load in list_models():
        splits = args.load.split('/')
        name_test = splits[-4] + "_" + splits[-1].split(".")[0]
    else:
        name_test = args.load


    with open(os.path.join(log_dir, f"{args.pos_subset}_{name_test}_{args.dataset}_ooo_inv{args.rotation}.csv"), "w") as f:
        wcsv = csv.writer(f)
        acc = invariance_max(args)
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
    parser.add_argument('--load', default="videomae", type=str)
    parser.add_argument('--model', default="resnet50", type=str)
    parser.add_argument('--head', default="", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--load_strict', default=True, type=str2bool)
    parser.add_argument('--rotation', default="", type=str)
    parser.add_argument('--keep_proj', default=["projector"], type=str2table)
    parser.add_argument('--dense_features', default=False, type=str2bool)

    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--num_devices', default=1, type=int)

    args = parser.parse_args()
    args.dataset = "shepardmetzler"
    args.pos_subset = "rotated"
    args.neg_subset = "mirror"
    # assert args.head == "action_prediction", "Need action prediction module"
    args.log_dir = os.path.join(args.log_dir, args.dataset, "invmax")
    if args.dense_features:
        args.log_dir += "dense"
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    start_invariance_max(args, args.log_dir)

