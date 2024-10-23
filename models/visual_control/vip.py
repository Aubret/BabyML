import os

from configs.Misc.mmdet_mask_rcnn_R_50_FPN_1x import model
from torch.utils import model_zoo
from torchvision.models import resnet50

#Adapted from https://github.com/facebookresearch/r3m


def VIP_Model():
    modelurl = "https://pytorch.s3.amazonaws.com/models/rl/vip/model.pt"
    model = resnet50()

    r3m_state_dict = model_zoo.load_url(modelurl)['vip']
    model.load_state_dict(r3m_state_dict)
    return model
