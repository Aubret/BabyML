import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/..")

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import config
from lightning.fabric.strategies import DDPStrategy
import lightning as L

from datasets import DATASETS
from models.heads import ActionMultiLayerProj, MultiLayerProj, MultiLayerProjShortcut
from tools import BACKBONES, load_model, get_transforms, add_head, get_features


@torch.no_grad()
def mental_rotation(args):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.seed_everything(args.seed)
    fabric.launch()

    preprocess = get_transforms(args.dataset)
    model = BACKBONES[args.model]()
    add_head(model, args.head)
    # n_out = 2048
    n_out = 512
    model.head_action = ActionMultiLayerProj(2, 2 * n_out, 2048, 128, bias=False)
    # model.head_equivariant = MultiLayerProj(1, 2048, 4096, 128, bias=False)
    model.head_prediction = MultiLayerProjShortcut(2, n_out+128, 4096, n_out, bias=False)
    # model.head_prediction = MultiLayerProj(2, 256, 4096, 128, bias=False)
    if args.load != "random":
        checkpoint = torch.load(args.load, map_location="cpu")
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        to_remove = []
        for k in checkpoint.keys():
            if k.startswith("fc."):
                to_remove.append(k)
            if k.startswith("classifier."):
                to_remove.append(k)

        for k in to_remove:
            del checkpoint[k]
        model.load_state_dict(checkpoint, strict=args.load_strict)

    dataset = DATASETS[args.dataset](args.data_root, subset_name="features", transform=preprocess)
    dataset_pos = DATASETS[args.dataset](args.data_root, subset_name=args.pos_subset, transform=preprocess)
    dataset_neg = DATASETS[args.dataset](args.data_root, subset_name=args.neg_subset, transform=preprocess)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_pos = DataLoader(dataset_pos, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_neg = DataLoader(dataset_neg, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)

    dataloader, dataloader_pos, dataloader_neg = fabric.setup_dataloaders(dataloader, dataloader_pos, dataloader_neg)



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
    naive_acc= ((correct >= wrong1) & (correct >= wrong2)).float().mean().item()

    naive_acc_proj, mental_acc1, mental_acc2, cpt, d1, d2, d3, d4, d5, d6 = 0, 0, 0, 0, 0, 0, 0, 0, correct.mean().item(), wrong1.mean().item()
    for f, f_p, f_n in zip(features.split(64), features_pos.split(64), features_neg.split(64)):
        pred_action = model.head_action.forward_all(f, f_p)
        pred_action_n = model.head_action.forward_all(f, f_n)

        if hasattr("model", "head_equivariant"):
            proj, proj_p, proj_n = model.head_equivariant(f),model.head_equivariant(f_p),model.head_equivariant(f_n)
        else:
            proj, proj_p, proj_n = f, f_p, f_n
        pred_proj = model.head_prediction(torch.cat((proj, pred_action), dim=1))
        pred_proj_n = model.head_prediction(torch.cat((proj, pred_action_n), dim=1))

        correct = torch.nn.functional.cosine_similarity(pred_proj, proj_p, dim=1)
        wrong1 = torch.nn.functional.cosine_similarity(pred_proj_n, proj_n, dim=1)
        wrong2 = torch.nn.functional.cosine_similarity(pred_proj, proj, dim=1)
        wrong3 = torch.nn.functional.cosine_similarity(pred_proj_n, proj, dim=1)
        mental_acc1 += (correct >= wrong1).float().sum().item()
        mental_acc2 += ((correct >= wrong1) & (correct >= wrong2) & (correct >= wrong3)).float().sum().item()
        d1 += correct.sum().item()
        d2 += wrong1.sum().item()
        d3 += wrong2.sum().item()
        d4 += wrong3.sum().item()
        cpt += len(f)

        c = torch.nn.functional.cosine_similarity(proj, proj_p, dim=1)
        w1 = torch.nn.functional.cosine_similarity(proj, proj_n, dim=1)
        w2 = torch.nn.functional.cosine_similarity(proj_p, proj_n, dim=1)
        naive_acc_proj += ((c >= w1) & (c >= w2)).float().sum().item()

    return naive_acc, naive_acc_proj/cpt, mental_acc1/cpt, mental_acc2/cpt, d1/cpt, d2/cpt, d3/cpt, d4/cpt, d5, d6


if __name__ == '__main__':
    config.parser.add_argument('--pos_subset', default="rotated", type=str)
    config.parser.add_argument('--neg_subset', default="mirror", type=str)
    args = config.parser.parse_args()
    assert args.dataset == "shepardmetzler", "only work with ShepardMetzler dataset"
    # assert args.head == "action_prediction", "Need action prediction module"
    args.log_dir = os.path.join(args.log_dir, args.dataset, "mental")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


    name_test = args.load.split('/')[-1].split(".")[0]
    with open(os.path.join(args.log_dir, f"{args.pos_subset}_{name_test}_{args.seed}_{args.dataset}_ooo_mental.csv"), "w") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["naive", "naive_proj", "mr", "mr2", "dmr1", "dmr2", "dmr3", "dmr4", "dmr5", "dmr6"])
        acc = [*mental_rotation(args)]
        wcsv.writerow(acc)
