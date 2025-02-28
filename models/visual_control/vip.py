import os

import gdown
import torch
from torch import nn
from torchvision.models import resnet50


#Adapted from https://github.com/facebookresearch/vip


def VIP_Model():
    modelurl = "https://pytorch.s3.amazonaws.com/models/rl/vip/model.pt"
    model = resnet50()
    model.fc = nn.Identity()

    if not os.path.exists(f"{os.environ['HOME']}/.cache/torch/hub/checkpoints/"):
        os.makedirs(f"{os.environ['HOME']}/.cache/torch/hub/checkpoints/")

    home = f"{os.environ['HOME']}/.cache/torch/hub/checkpoints/vip.pt"
    if not os.path.exists(home):
        gdown.download(modelurl, home, quiet=False)
    vip_state_dict = torch.load(home)['vip']
    # r3m_state_dict = model_zoo.load_url(modelurl)['vip']
    model.fc = nn.Identity()
    model.load_state_dict(rename_layers(vip_state_dict))
    return model

def rename_layers(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if "fc" in k:
            continue
        k2 = ".".join(k.split(".")[2:])
        new_state_dict[k2] = v
    return new_state_dict