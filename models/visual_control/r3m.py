import os

from configs.Misc.mmdet_mask_rcnn_R_50_FPN_1x import model
from torch.utils import model_zoo
from torchvision.models import resnet50

#Adapted from https://github.com/facebookresearch/r3m


def R3M_Model():
    # foldername = "r3m_50"
    modelurl = 'https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA'
    # configurl = 'https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8'
    model = resnet50()

    r3m_state_dict = remove_language_head(model_zoo.load_url(modelurl)['r3m'])
    model.load_state_dict(r3m_state_dict)
    return model


def remove_language_head(state_dict):
    keys = state_dict.keys()
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key in list(keys):
        if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]
    return state_dict
