import argparse
import csv
import math
import os
import re
import sys

import numpy as np
import scipy
import torchvision.utils

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
        if re.search("^projector.*", k):
            continue
        if re.search("^sup_lin*", k):
            continue
        # if "action_projector" in k and not "action_projector" in args.keep_proj:
        #     continue
        if "action_head" in k:
            if "action_head" in args.keep_proj:
                new_k = ".".join(["action.layers"] + k.split(".")[3:])
                new_state_dict[new_k] = w
        elif "action_projector" in k:
            if "action_projector" in args.keep_proj:
                new_k = ".".join(["head_action.layers"] + k.split(".")[2:])
                new_state_dict[new_k] = w
        elif "action_predictor" in k:
            if "action_predictor" in args.keep_proj:
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

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def relative_quaternion(a0, a1):
    eul0 = scipy.spatial.transform.Rotation.from_euler("yzx", angles=np.array([a0,0,0]), degrees=True)
    eul1 = scipy.spatial.transform.Rotation.from_euler("yzx", angles=np.array([a1,0,0]), degrees=True)
    translation0 = np.array([math.cos(math.pi * a0/180), math.sin(math.pi* a0/180), 0])
    translation1 = np.array([math.cos(math.pi* a1/180), math.sin(math.pi* a1/180), 0])

    a0 = scipy.spatial.transform.Rotation.as_quat(eul0)
    a1 = scipy.spatial.transform.Rotation.as_quat(eul1)
    q0 = np.concatenate((a0[1:4], a0[0:1]), axis=0)
    q1 = np.concatenate((a1[1:4], a1[0:1]), axis=0)
    # q0, q1 = a0, a1

    resq = quaternion_multiply((q0[3:4], -q0[0:1], -q0[1:2], -q0[2:3]), (q1[3:4], q1[0:1], q1[1:2], q1[2:3])).squeeze()

    translation = translation1 - translation0
    # r0 = scipy.spatial.transform.Rotation.from_quat(q0)
    # translation = np.matmul(np.transpose(r0.as_matrix()), translation)


    action = torch.tensor(np.concatenate((resq, np.clip(translation, -50, 50), np.array([0])), axis=0), dtype=torch.float32)

    action_mean = torch.tensor([0.759, -0.000354, -0.00682, -0.00723, 0.00314, 0.00787, -0.0171, 0])
    action_std = torch.tensor([0.431, 0.048, 0.358, 0.328, 3.8, 1.25, 1.13, 1])
    action = (action - action_mean)/action_std
    return action

def relative_quaternion2(a):
    eul = scipy.spatial.transform.Rotation.from_euler("yzx", angles=np.array([a,0,0]), degrees=True)
    a = scipy.spatial.transform.Rotation.as_quat(eul)
    a = np.concatenate((a[1:4], a[0:1]), axis=0)
    # a = np.concatenate((a[3:4], a[0:3]), axis=0)

    action_mean = torch.tensor([0.759, -0.000354, -0.00682, -0.00723])
    action_std = torch.tensor([0.431, 0.048, 0.358, 0.328])
    action = (torch.tensor(a)- action_mean)/action_std
    return torch.cat((action, torch.tensor([0,0,0,0])))

def inverse_transform(pred_action):
    action_mean = torch.tensor([0.759, -0.000354, -0.00682, -0.00723, 0.00314, 0.00787, -0.0171, 0])
    action_std = torch.tensor([0.431, 0.048, 0.358, 0.328, 3.8, 1.25, 1.13, 1])

    pred_action = pred_action.cpu() * action_std + action_mean
    pred_action = pred_action[:4].numpy()
    pred_action = scipy.spatial.transform.Rotation.from_quat(pred_action)
    return pred_action.as_euler("yzx", degrees=True)


@torch.no_grad()
def predict_action(args):
    action = relative_quaternion(100, 150).to("cuda:0")
    action = relative_quaternion2(50).to("cuda:0")
    # print(action, relative_quaternion2(50))
    # print(action)

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
    # model.head_action = ActionMultiLayerProj(2, 2 * n_out, 2048, n_features, bias=False)
    # model.head_prediction = MultiLayerProjShortcut(2, n_out+n_features, 4096, n_out, bias=False)
    # print(model)

    # model.head_action = ActionMultiLayerProj(2, 2 * n_out, 2048, n_features, bias=False)
    model.head_action = ActionMultiLayerProj(2, 2 * n_out, 1024, 8, bias=True)
    # model.action = MultiLayerProj(1, 8, 2048, n_features, bias=False)
    # model.action = torch.nn.Identity()
    model.ciper_action_bn = torch.nn.BatchNorm1d(8, affine=False)

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
    whitebg=False
    dataset = DATASETS[args.dataset](args.data_root, subset_name=args.pos_subset, transform=preprocess, whitebg=whitebg, rotation="100")
    dataset_pos = DATASETS[args.dataset](args.data_root, subset_name=args.pos_subset, transform=preprocess, whitebg=whitebg, rotation="150")
    dataset_neg = DATASETS[args.dataset](args.data_root, subset_name=args.neg_subset, transform=preprocess, whitebg=whitebg,rotation="150")

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_pos = DataLoader(dataset_pos, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_neg = DataLoader(dataset_neg, batch_size=16, shuffle=False, pin_memory=True, num_workers=1)

    dataloader, dataloader_pos, dataloader_neg = fabric.setup_dataloaders(dataloader, dataloader_pos, dataloader_neg)

    # for img, label, img_id in dataloader:
    #     for i, idi in zip(img, img_id):
    #         torchvision.utils.save_image(i, "/home/fias/postdoc/gym_results/test_images/sheperdwhite/"+str(idi.item())+".png")
    #     break
    # for img, label, img_id in dataloader_pos:
    #     for i, idi in zip(img, img_id):
    #         torchvision.utils.save_image(i, "/home/fias/postdoc/gym_results/test_images/sheperdwhite2/"+str(idi.item())+".png")
    #     break
    # return
    model = fabric.setup(model)
    model.eval()

    features, labels, img_ids = get_features(dataloader, model, fabric)
    features_pos, labels_pos, img_ids_pos = get_features(dataloader_pos, model, fabric)
    features_neg, labels_neg, img_ids_neg = get_features(dataloader_neg, model, fabric)

    # assert to verify the shuffle=False works well
    assert (img_ids_neg == img_ids_pos).float().sum() == img_ids_pos.shape[0]


    correct = torch.nn.functional.cosine_similarity(features, features_pos, dim=1)
    wrong1 = torch.nn.functional.cosine_similarity(features, features_neg, dim=1)
    # wrong2 = torch.nn.functional.cosine_similarity(features_pos, features_neg, dim=1)
    # naive_acc= ((correct >= wrong1) & (correct >= wrong2)).float().mean().item()
    naive_acc= (correct > wrong1).float().mean().item()

    action_acc, action_acc2, cpt = 0, 0, 0
    action = model.action(action.unsqueeze(0)) if not "ciper" in args.load else model.ciper_action_bn(action.unsqueeze(0))
    for f, f_p, f_n in zip(features.split(64), features_pos.split(64), features_neg.split(64)):
        if hasattr(model, "action_rep_projector"):
            f, f_p, f_n = model.action_rep_projector(f),model.action_rep_projector(f_p),model.action_rep_projector(f_n)

        pred_action = model.head_action.forward_all(f, f_p)
        pred_action_n = model.head_action.forward_all(f, f_n)

        # print(action, pred_action[0].view(-1))

        action_acc += (torch.norm(pred_action - action, dim=1) < torch.norm(pred_action_n - action, dim=1)).float().sum().item()
        cpt += f.shape[0]

        if "ciper" in args.load:
            pred_action = pred_action * (model.ciper_action_bn.running_var ** 0.5) + model.ciper_action_bn.running_mean
            pred_action_n = pred_action_n * (model.ciper_action_bn.running_var ** 0.5) + model.ciper_action_bn.running_mean
            for i in range(f.shape[0]):
                # print(inverse_transform(pred_action[i]), inverse_transform(pred_action_n[i]))
                action_acc2 += float(abs(inverse_transform(pred_action[i])[0] - 50) < abs(inverse_transform(pred_action_n[i])[0] - 50))


    print(naive_acc, action_acc/cpt, action_acc2/cpt)
    return naive_acc, action_acc/cpt, action_acc2/cpt


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
    parser.add_argument('--keep_proj', default=["action_projector","action_head","action_predictor"], type=str2table)

    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--num_devices', default=1, type=int)

    args = parser.parse_args()
    args.pos_subset = "rotated"
    args.neg_subset = "mirror"
    args.dataset = "shepardmetzler"
    # assert args.head == "action_prediction", "Need action prediction module"
    args.log_dir = os.path.join(args.log_dir, args.dataset, "action")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    splits = args.load.split('/')
    name_test = splits[-4]+"_"+splits[-1].split(".")[0]
    with open(os.path.join(args.log_dir, f"{args.pos_subset}_{name_test}_{args.dataset}_ooo_action.csv"), "w") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["naive", "action_pred"])
        acc = [*predict_action(args)]
        wcsv.writerow(acc)
