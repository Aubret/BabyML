
from torch import nn
from torchvision.models import resnet18
from torchvision.models import resnet50

from . import model_zoo
from .registry import list_models


def resnet18_default(*args, **kwargs):
    model = resnet18(*args, **kwargs)
    model.fc = nn.Identity()
    return model


def resnet50_default( *args, **kwargs):
    model = resnet50(*args, **kwargs)
    model.fc = nn.Identity()
    return model

