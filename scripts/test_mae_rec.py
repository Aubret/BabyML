import argparse
import os
import re
import sys

import torchvision.utils
from functorch.einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/..")
from models.mae.videomae_pretrain import PretrainVisionTransformer, pretrain_videomae_base_patch16_224

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from lightning.fabric.strategies import DDPStrategy
import lightning as L
from datasets import DATASETS
from models.heads import ActionMultiLayerProj, MultiLayerProj, MultiLayerProjShortcut
from tools import BACKBONES, load_model, get_transforms, add_head, get_features
from torchvision.transforms import v2 as trv2, InterpolationMode
import numpy as np
from collections import OrderedDict
import gdown

model_urls = {
    'something': 'https://drive.google.com/uc?id=1I18dY_7rSalGL8fPWV82c0-foRUDzJJk',
    "ucf":'https://drive.google.com/uc?id=1BHev4meNgKM0o_8DMRbuzAsKSP3IpQ3o',
    "kinetic": "https://drive.google.com/uc?id=1qLOXWb_MGEvaI7tvuAe94CV7S2HXRwT3",
    "something_finetune": "https://drive.google.com/uc?id=1dt_59tBIyzdZd5Ecr22lTtzs_64MOZkT"
}


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

def load(model, name_url="something"):
    home = os.environ['HOME']
    name = f"videomae.ckpt" if name_url == "something" else f"videomae_{name_url}.ckpt"
    if not os.path.exists(f"{home}/.cache/torch/hub/checkpoints/"):
        os.makedirs(f"{home}/.cache/torch/hub/checkpoints/")

    if not os.path.exists(f"{os.environ['HOME']}/.cache/torch/hub/checkpoints/{name}"):
        gdown.download(model_urls[name_url], f'{home}/.cache/torch/hub/checkpoints/{name}', quiet=False)

    checkpoint = torch.load(f'/home/fias/.cache/torch/hub/checkpoints/{name}', map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    return model

def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Patchifies an image according to some patch size.
    Adapted from https://github.com/facebookresearch/mae.

    Args:
        imgs (torch.Tensor): [N, 3, H, W] Tensor containing the original images.
        patch_size (int): size of each patch.

    Returns:
        torch.Tensor: [N, Tokens, pixels * pixels * 3] Tensor containing the patchified images.
    """
    # print(imgs.size(2), imgs.size(3))
    assert imgs.size(-2) == imgs.size(-1) and imgs.size(-2) % patch_size == 0

    h = w = imgs.size(2) // patch_size
    x = imgs.reshape(shape=(imgs.size(0), 3, h, patch_size, w, patch_size))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.size(0), h * w, patch_size**2 * 3))
    return x

@torch.no_grad()
def reconstruction(args):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.launch()

    # mean, std, image_size = (0, 0, 0), (1, 1, 1), 224
    mean, std, image_size =  (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
    preprocess = trv2.Compose([trv2.Resize(image_size, interpolation=InterpolationMode.BICUBIC), trv2.CenterCrop(image_size),
    # preprocess = trv2.Compose([trv2.Resize(image_size), trv2.CenterCrop(image_size),
                         trv2.ToImage(), trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])


    n_frame = 8
    # if args.load in list_models():
    #     model = model_registry[args.load](n_frame=n_frame)
    #     if isinstance(model, tuple) and len(model) > 1:
    #         model, preprocess = model
    # else:
    #     model = BACKBONES[args.model]()
    #     model = complete_head(model)
    model = pretrain_videomae_base_patch16_224(decoder_depth=4, num_frames = n_frame)
    model = load(model, "something")
    # model = vit_base_patch16_224(pretrained=False, drop_path_rate=0, use_mean_pooling=True, all_frames=n_frame)

    whitebg=False
    dataset = DATASETS[args.dataset](args.data_root, subset_name=args.pos_subset, transform=preprocess, whitebg=whitebg, rotation=["0","50","100","150"])
    dataset_pos = DATASETS[args.dataset](args.data_root, subset_name=args.pos_subset, transform=preprocess, whitebg=whitebg, rotation=["150"])
    dataset_neg = DATASETS[args.dataset](args.data_root, subset_name=args.neg_subset, transform=preprocess, whitebg=whitebg, rotation=["150"])

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_pos = DataLoader(dataset_pos, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)
    dataloader_neg = DataLoader(dataset_neg, batch_size=32, shuffle=False, pin_memory=True, num_workers=1)

    dataloader, dataloader_pos, dataloader_neg = fabric.setup_dataloaders(dataloader, dataloader_pos, dataloader_neg)

    model = fabric.setup(model)
    model.eval()
    #
    # for img, label, img_id in dataloader:
    #     print(img.shape)
    #     for i, idi in zip(img, label):
    #         torchvision.utils.save_image(i, "/home/fias/postdoc/gym_results/test_images/sheperdblack/"+str(idi.item())+".png")
    #     break
    # for img, label, img_id in dataloader_pos:
    #     for i, idi in zip(img, label):
    #         torchvision.utils.save_image(i, "/home/fias/postdoc/gym_results/test_images/sheperdblack2/"+str(idi.item())+".png")
    #     break
    # for img, label, img_id in dataloader_neg:
    #     for i, idi in zip(img, label):
    #         torchvision.utils.save_image(i, "/home/fias/postdoc/gym_results/test_images/sheperdblack3/"+str(idi.item())+".png")
    #     break
    # return


    imgs, labels, img_ids, rotations = [], [], [], []
    imgsp, labelsp, img_idsp = [], [], []
    imgsn, labelsn, img_idsn = [], [], []

    for r in tqdm(dataloader):
        imgs.append(r[0])
        labels.append(r[1])
        img_ids.append(r[2])
        rotations.append(r[3])
    # for rp in tqdm(dataloader_pos):
    #     imgsp.append(rp[0])
    #     labelsp.append(rp[1])
    #     img_idsp.append(rp[2])
    #
    for rn in tqdm(dataloader_neg):
        imgsn.append(rn[0])
        labelsn.append(rn[1])
        img_idsn.append(rn[2])

    imgs = torch.cat(imgs)
    # imgsp = torch.cat(imgsp)
    imgsn = torch.cat(imgsn)


    labels = torch.cat(labels)
    rotations = torch.cat(rotations)
    # labelsp = torch.cat(labelsp)
    labelsn = torch.cat(labelsn)

    features, featuresp, featuresn = [], [], []
    features1, features2, features3 = [], [], []

    mean = torch.as_tensor(mean).to("cuda:0")[None, :, None, None]
    std = torch.as_tensor(std).to("cuda:0")[None, :, None, None]
    for id in range(1, 49, 1):
        mask = labels == id
        # maskp = labelsp == id
        # maskn = labelsn == id

        inputsm = imgs[mask]
        arrange = torch.argsort(rotations[mask])
        inputsm = inputsm[arrange]

        inputs = inputsm.unsqueeze(1).repeat(1,n_frame // inputsm.shape[0],1,1,1).flatten(start_dim=0,end_dim=1)
        inputs = inputs.permute(1,0,2,3).unsqueeze(0)

        mask_in = torch.zeros((inputs.shape[0], 196 * 3 + 98), dtype=torch.bool)
        mask_out = torch.ones((inputs.shape[0], 98), dtype=torch.bool)
        mask_all = torch.cat((mask_in, mask_out), dim=1)
        # features.append(model(inputs, nrepeat=1, mask=mask_all))
        out = model(inputs, nrepeat=1, mask=mask_all)

        # featuresp.append(model(imgsp[maskp], nrepeat=n_frame, mask=torch.zeros((1, 196*4), dtype=torch.bool)))
        # featuresn.append(model(imgsn[maskn], nrepeat=n_frame, mask=torch.zeros((1, 196*4), dtype=torch.bool)))
        # pix = patchify(imgs[-2:-1], 16)
        # pix= torch.stack((imgs[-2:-1], imgs[-2:-1]), 2).flatten(2)

        # out_save = out[:,out.shape[1]//2].view(1,-1)
        # rec_image = out*std + mean
        # torchvision.utils.save_image(rec_image, "/home/fias/gym_results/test_images/sheperdblack4/"+str(id)+".png")
         # print(pix.shape, out.shape)

        # img_squeeze = rearrange(inputsm[-3:-2].unsqueeze(2), 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=1, p1=16, p2=16)
        # ori_img = inputsm[-2:-1] * std + mean
        # torchvision.utils.save_image(ori_img.clamp(0,0.996), "/home/fias/postdoc/gym_results/test_images/sheperdblack5/"+str(id)+".png")

        # out = (out - out.mean(dim=(0,1,2),keepdim=True)) / ((out.var(dim=(0,1,2), keepdim=True) + 1.0e-6) ** 0.5)
        # out = (out - out.mean(keepdim=True)) / ((out.var(keepdim=True) + 1.0e-6) ** 0.5)

        img_squeeze = rearrange(inputs, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=16, p2=16)
        img_patch = rearrange(img_squeeze, 'b n p c -> b n (p c)')
        img_patch[mask_all] = out

        # print(out.shape)
        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
        rec_img = rec_img[:,-196:] * (img_squeeze[:,-196:].var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        # rec_img = rec_img[:,-196:]  #(img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
        rec_img = rearrange(rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=16, p2=16, h=14, w=14)
        rec_img = rec_img[:,:,0]# * std + mean
        torchvision.utils.save_image(rec_img.clamp(0,0.996), "/home/fias/postdoc/gym_results/test_images/sheperdblack4/"+str(id)+".png")


    # features = torch.cat(features)
    # featuresp = torch.cat(featuresp)
    # featuresn = torch.cat(featuresn)
    # #
    # correct = torch.nn.functional.cosine_similarity(features, featuresp, dim=1)
    # # correct = -torch.norm(features - featuresp, dim=1)
    # wrong = torch.nn.functional.cosine_similarity(features, featuresn, dim=1)
    # # wrong = -torch.norm(features - featuresn, dim=1)
    # acc= [(correct > wrong).float().sum().item()]
    #
    # if len(features1) > 0:
    #     features_all = torch.stack((torch.cat(features1), torch.cat(features2), torch.cat(features3)), dim=1)
    #     correct_all = torch.nn.functional.cosine_similarity(featuresp.unsqueeze(1), features_all, dim=2)
    #     wrong_all = torch.nn.functional.cosine_similarity(featuresn.unsqueeze(1), features_all, dim=2)
    #
    #     # correct_all = -torch.norm(featuresp.unsqueeze(1) - features_all, dim=2)
    #     # wrong_all = -torch.norm(featuresn.unsqueeze(1) - features_all, dim=2)
    #
    #     corrects = torch.max(correct_all, dim=1).values
    #     wrongs = torch.max(wrong_all, dim=1).values
    #     acc.append((corrects > wrongs).float().sum().item())



    #
    return np.array(acc)/features.shape[0]


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
    parser.add_argument('--load', default="videomae", type=str)
    parser.add_argument('--model', default="resnet50", type=str)
    parser.add_argument('--head', default="", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--load_strict', default=True, type=str2bool)
    parser.add_argument('--rotation', default="", type=str)
    parser.add_argument('--keep_proj', default=["projector"], type=str2table)

    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--num_devices', default=1, type=int)

    args = parser.parse_args()
    args.dataset = "shepardmetzler"
    args.pos_subset = "rotated"
    args.neg_subset = "mirror"
    # assert args.head == "action_prediction", "Need action prediction module"
    # args.log_dir = os.path.join(args.log_dir, args.dataset, "invmulti")
    # if not os.path.exists(args.log_dir):
    #     os.makedirs(args.log_dir)

    reconstruction(args)

