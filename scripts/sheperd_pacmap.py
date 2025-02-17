import argparse
import csv
import os
import re
import sys

import pacmap
import torchvision.utils
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/..")

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from lightning.fabric.strategies import DDPStrategy
import lightning as L

from datasets import DATASETS
from models.heads import ActionMultiLayerProj, MultiLayerProj, MultiLayerProjShortcut
from tools import BACKBONES, load_model, get_transforms, add_head, get_features
from torchvision.transforms import v2 as trv2, InterpolationMode

def get_pacmap(representations, labels, class_labels):
    """
        Draw the PacMAP plot
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
            epoch: epoch (int)
        return:
            fig: the PacMAP plot (matplotlib.figure.Figure)
    """
    # sns.set()
    sns.set_style("ticks")
    sns.set_context('paper', font_scale=1.8, rc={'lines.linewidth': 2})
    color_map = ListedColormap(sns.color_palette('colorblind', 50))
    legend_patches = [Patch(color=color_map(i), label=label) for i, label in enumerate(class_labels)]
    # save the visualization result
    embedding = pacmap.PaCMAP(2)
    X_transformed = embedding.fit_transform(representations.cpu().numpy(), init="pca")
    fig, ax = plt.subplots(1, 1, figsize=(7.7,4.8))

    # labels = labels.cpu().numpy()
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels, cmap=color_map, s=20)
    ax.set_title("pacmap")
    plt.xticks([]), plt.yticks([])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1.), handles=legend_patches, fontsize=13.8)
    return fig

def clean_checkpoint(checkpoint):
    new_state_dict = {}
    for k, w in checkpoint.items():

        if re.search("^model.*", k):
            k = ".".join(k.split(".")[1:])
        if re.search("^projector.*", k):
            continue
        if re.search("^sup_lin*", k):
            continue
        # if "action_projector" in k and not "action_projector" in args.keep_proj:
        #     continue
        if "action_head" in k and not "action_head" in args.keep_proj:
            continue

        if "action_projector" in k and "action_projector" in args.keep_proj:
            new_k = ".".join(["head_action.layers"] + k.split(".")[2:])
            new_state_dict[new_k] = w
        elif "equivariant_projector" in k and "equivariant_projector" in args.keep_proj:
            new_k = ".".join(["head_equivariant.layers"] + k.split(".")[2:])
            new_state_dict[new_k] = w
        elif "equivariant_predictor" in k and "equivariant_predictor" in args.keep_proj:
            new_k = ".".join(["head_prediction.layers"] + k.split(".")[2:])
            new_state_dict[new_k] = w
        elif "action_rep_projector" in k:
            new_k = k.replace("net","layers")
            new_state_dict[new_k] = w
        else:
            new_state_dict[k] = w
    return new_state_dict

@torch.no_grad()
def mental_rotation(args):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.launch()

    # mean, std, image_size = (0, 0, 0), (1, 1, 1), 224
    mean, std, image_size =  (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
    preprocess = trv2.Compose([trv2.Resize(image_size, interpolation=InterpolationMode.BICUBIC), trv2.CenterCrop(image_size),
                         trv2.ToImage(), trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])

    model = BACKBONES[args.model]()
    add_head(model, args.head)
    # n_out = 2048
    n_out = 512 if args.model == "resnet18" else 2048
    n_features = 128
    print(n_out)
    model.head_action = ActionMultiLayerProj(2, 2 * n_out, 2048, n_features, bias=False)
    model.head_prediction = MultiLayerProjShortcut(2, n_out+n_features, 4096, n_out, bias=False)
    # print(model)

    # n_features = 256
    # model.head_action = ActionMultiLayerProj(2, 2 * n_out, 2048, n_features, bias=False)
    # model.head_prediction = MultiLayerProj(2, n_out+n_features, 4096, n_out, bias=False)

    # model.head_action = ActionMultiLayerProj(2, 2 * n_features, 2048, 128, bias=False)
    # model.action_rep_projector = MultiLayerProj(2, n_out, 2048, 128, bias=False)
    # model.head_prediction = MultiLayerProjShortcut(2, 2*n_features, 4096, 128, bias=False)

    # model.head_equivariant = MultiLayerProj(1, 2048, 4096, n_features, bias=False)
    # model.head_prediction = MultiLayerProj(2, 256, 4096, n_features, bias=False)
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

    # print(preprocess)
    dataset = DATASETS[args.dataset](args.data_root, subset_name="features", transform=preprocess)
    dataset_pos = DATASETS[args.dataset](args.data_root, subset_name=args.pos_subset, transform=preprocess)
    dataset_neg = DATASETS[args.dataset](args.data_root, subset_name=args.neg_subset, transform=preprocess)

    dataloader_pos = DataLoader(dataset_pos, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_neg = DataLoader(dataset_neg, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)

    dataloader_pos, dataloader_neg = fabric.setup_dataloaders(dataloader_pos, dataloader_neg)

    model = fabric.setup(model)
    model.eval()

    features_pos, labels_pos, img_ids_pos = get_features(dataloader_pos, model, fabric)
    features_neg, labels_neg, img_ids_neg = get_features(dataloader_neg, model, fabric)

    all_features = torch.cat((features_pos[:4], features_neg[:4]), dim=0 )
    labels = [0,0,0,0,1,1,1,1]
    fig = get_pacmap(all_features, labels, [0,1])
    fig.savefig("/home/fias/postdoc/gym_results/test_images/pacmap/mrtest.png")


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
    parser.add_argument('--data_root', default="data", type=str)
    parser.add_argument('--log_dir', default="logs", type=str)
    parser.add_argument('--load', default="random", type=str)
    parser.add_argument('--model', default="resnet50", type=str)
    parser.add_argument('--head', default="", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--load_strict', default=True, type=str2bool)
    parser.add_argument('--keep_proj', default=[], type=str2table)

    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--num_devices', default=1, type=int)

    parser.add_argument('--pos_subset', default="rotated", type=str)
    parser.add_argument('--neg_subset', default="mirror", type=str)
    args = parser.parse_args()
    args.dataset = "shepardmetzler"
    # assert args.head == "action_prediction", "Need action prediction module"
    args.log_dir = os.path.join(args.log_dir, args.dataset, "mental")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    splits = args.load.split('/')
    name_test = splits[-4]+"_"+splits[-1].split(".")[0]
    mental_rotation(args)
