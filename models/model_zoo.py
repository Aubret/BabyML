import clip
import torch
import torchvision
from torchvision.models import ViT_L_16_Weights

from models.byol.byol import Byol
from models.mae.mae import MAE
from models.mvimgnet.ssltt import mvimgnet
from models.visual_control.r3m import R3M_Model
from models.visual_control.vc1 import VC1_Model
from models.visual_control.vip import VIP_Model
from registry import register_model


def model_pytorch(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__[model_name](pretrained=True)
    return model


@register_model()
def Clip():
    model, preprocess = clip.load("ViT-L/14", device="cpu")
    model.forward = model.encode_image
    return model, preprocess

@register_model()
def ClipRN50():
    model, preprocess = clip.load("RN50")
    model.forward = model.encode_image
    return model, preprocess

@register_model()
def SupViTL16():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    return model

@register_model()
def DinoV2():
    return torchvision.models.vit_l_16(weights =ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)


@register_model()
def ByolRN50():
    return Byol("rn50")

@register_model()
def MAE_B():
    return MAE("vitb")

@register_model()
def MAE_L():
    return MAE("vitl")

@register_model()
def R3M():
    return R3M_Model()

@register_model()
def VIP():
    return VIP_Model()

@register_model()
def VC1():
    return VC1_Model()

# @register_model()
# def ByolRN200x4():
#     return Byol("rn200x4")

@register_model()
def AASimCLR():
    return mvimgnet("aasimclr")

@register_model()
def SimCLRmv():
    return mvimgnet("simclr")

@register_model()
def SimCLRTT():
    return mvimgnet("simclrtt")

@register_model()
def CiperSimCLR():
    return mvimgnet("cipersimclr")

@register_model()
def MoCo():
    from .pycontrast.pycontrast_resnet50 import MoCo
    model, classifier = MoCo(pretrained=True)
    return model


@register_model()
def MoCoV2():
    from .pycontrast.pycontrast_resnet50 import MoCoV2
    model, classifier = MoCoV2(pretrained=True)
    return model

@register_model()
def bagnet9():
    from .bagnets.bagnets import bagnet9
    return bagnet9(pretrained=True)


@register_model()
def bagnet17():
    from .bagnets.bagnets import bagnet17
    return bagnet17(pretrained=True)


@register_model()
def bagnet33():
    from .bagnets.bagnets import bagnet33
    return bagnet33(pretrained=True)


@register_model()
def resnet50_l2_eps0():
    from .adversarially_robust.robust_models import resnet50_l2_eps0
    return resnet50_l2_eps0()


@register_model()
def resnet50_l2_eps0_01():
    from .adversarially_robust.robust_models import resnet50_l2_eps0_01
    return resnet50_l2_eps0_01()


@register_model()
def resnet50_l2_eps0_03():
    from .adversarially_robust.robust_models import resnet50_l2_eps0_03
    return resnet50_l2_eps0_03()


@register_model()
def resnet50_l2_eps0_05():
    from .adversarially_robust.robust_models import resnet50_l2_eps0_05
    return resnet50_l2_eps0_05()


@register_model()
def resnet50_l2_eps0_1():
    from .adversarially_robust.robust_models import resnet50_l2_eps0_1
    return resnet50_l2_eps0_1()


@register_model()
def resnet50_l2_eps0_25():
    from .adversarially_robust.robust_models import resnet50_l2_eps0_25
    return resnet50_l2_eps0_25()

@register_model()
def resnet50_l2_eps0_5():
    from .adversarially_robust.robust_models import resnet50_l2_eps0_5
    return resnet50_l2_eps0_5()



@register_model()
def resnet50_l2_eps1():
    from .adversarially_robust.robust_models import resnet50_l2_eps1
    return resnet50_l2_eps1()



@register_model()
def resnet50_l2_eps3():
    from .adversarially_robust.robust_models import resnet50_l2_eps3
    return resnet50_l2_eps3()


@register_model()
def resnet50_l2_eps5():
    from .adversarially_robust.robust_models import resnet50_l2_eps5
    return resnet50_l2_eps5()

