import os

import gdown
import torch
from torch import nn
from torchvision.models import resnet50


#Adapted from https://github.com/facebookresearch/r3m


def R3M_Model():
    # foldername = "r3m_50"
    modelurl = 'https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA'
    # configurl = 'https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8'
    model = resnet50()
    model.fc = nn.Identity()
    if not os.path.exists(f"{os.environ['HOME']}/.cache/torch/hub/checkpoints/"):
        os.makedirs(f"{os.environ['HOME']}/.cache/torch/hub/checkpoints/")

    home = f"{os.environ['HOME']}/.cache/torch/hub/checkpoints/r3m.pt"
    if not os.path.exists(home):
        gdown.download(modelurl, home, quiet=False)
    r3m_state_dict = remove_language_head(torch.load(home)['r3m'])
    new_r3m_state_dict = rename_layers(r3m_state_dict)
    model.load_state_dict(new_r3m_state_dict)
    return model
    # r3m_state_dict = remove_language_head(model_zoo.load_url(modelurl)['r3m'])
    # model.load_state_dict(r3m_state_dict)
    # return model


def remove_language_head(state_dict):
    keys = state_dict.keys()
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key in list(keys):
        if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]
    return state_dict

def rename_layers(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        k2 = ".".join(k.split(".")[2:])
        new_state_dict[k2] = v
    return new_state_dict