from functools import partial

import clip
import open_clip
import timm
import torch
import torchvision
from timm.models import VisionTransformer
from torch import nn
from torchvision.models import ViT_L_16_Weights, resnet50, ResNet50_Weights

from models.byol.byol import Byol
from models.mae.mae import MAE
from models.mvimgnet.ssltt import mvimgnet
from models.visual_control.r3m import R3M_Model
from models.visual_control.vc1 import VC1_Model
from models.visual_control.vip import VIP_Model
from .jepa.vjepa import VJEPA
from .mae.omnimae import OmniMAE
from .mae.videomae import VideoMAE
from .registry import register_model
from torchvision.transforms import v2 as trv2, InterpolationMode

def model_pytorch(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__[model_name](pretrained=True)
    return model


@register_model()
def clip_vit():
    model, preprocess = clip.load("ViT-L/14")
    model.forward = model.encode_image
    return model, preprocess

@register_model()
def clip_rn50():
    model, preprocess = clip.load("RN50")
    model.forward = model.encode_image
    return model, preprocess

@register_model()
def dinov2():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    model.head = nn.Identity()
    return model

@register_model()
def sup_vitl16():
    model = torchvision.models.vit_l_16(weights =ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    model.head = nn.Identity()
    mean, std, image_size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 512
    preprocess = trv2.Compose([trv2.Resize(image_size, interpolation=InterpolationMode.BICUBIC), trv2.CenterCrop(image_size),
                  trv2.ToImage(), trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])

    return model, preprocess


@register_model()
def sup_vitl16classic():
    model = torchvision.models.vit_l_16(weights =ViT_L_16_Weights.IMAGENET1K_V1)
    model.head = nn.Identity()
    mean, std, image_size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
    preprocess = trv2.Compose([trv2.Resize(image_size, interpolation=InterpolationMode.BICUBIC), trv2.CenterCrop(image_size),
                  trv2.ToImage(), trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])

    return model, preprocess

@register_model()
def swsl_rn50():
    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnet50_swsl')
    model.fc = nn.Identity()
    return model

@register_model()
def swsl_rnx50():
    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnext101_32x16d_swsl')
    model.fc = nn.Identity()
    return model

@register_model()
def sup_rn50():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    return model

@register_model()
def byol_rn50():
    model = Byol("rn50")
    model.fc = nn.Identity()
    return model

@register_model()
def omnimae_vitb16(**kwargs):
    model = OmniMAE("vitb", **kwargs)
    model.head = nn.Identity()
    return model

@register_model()
def omnimae_vitl16(**kwargs):
    model = OmniMAE("vitl", **kwargs)
    model.head = nn.Identity()
    return model


@register_model()
def mae_vitb16():
    model = MAE("vitb")
    model.head = nn.Identity()
    return model

@register_model()
def mae_vitl16():
    model = MAE("vitl")
    model.head = nn.Identity()
    return model

@register_model()
def vjepa_vitl16(**kwargs):
    model = VJEPA("vitl",**kwargs)
    model.head = nn.Identity()
    return model

@register_model()
def vjepa_vith16(**kwargs):
    model = VJEPA("vith",**kwargs)
    model.head = nn.Identity()
    return model

@register_model()
def videomae(**kwargs):
    model = VideoMAE("something",**kwargs)
    model.head = nn.Identity()
    return model

@register_model()
def videomae_finetune(**kwargs):
    model = VideoMAE("something_finetune",**kwargs)
    model.head = nn.Identity()
    return model


@register_model()
def videomae_ucf(**kwargs):
    model = VideoMAE("ucf", **kwargs)
    model.head = nn.Identity()
    return model

@register_model()
def videomae_kinetic(**kwargs):
    model = VideoMAE("kinetic",**kwargs)
    model.head = nn.Identity()
    return model

@register_model()
def r3m():
    return R3M_Model()

@register_model()
def vip():
    return VIP_Model()

@register_model()
def vc1():
    return VC1_Model()

# @register_model()
# def ByolRN200x4():
#     return Byol("rn200x4")

@register_model()
def aasimclr():
    return mvimgnet("aasimclr")

@register_model()
def simclrmv():
    return mvimgnet("simclr")

@register_model()
def simclrtt():
    return mvimgnet("simclrtt")

@register_model()
def cipersimclr():
    return mvimgnet("cipersimclr")

@register_model()
def moco():
    from .pycontrast.pycontrast_resnet50 import MoCo
    model, classifier = MoCo(pretrained=True)
    model.fc = nn.Identity()
    return model


@register_model()
def mocov2():
    from .pycontrast.pycontrast_resnet50 import MoCoV2
    model, classifier = MoCoV2(pretrained=True)
    model.fc = nn.Identity()
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

@register_model()
def efficientnet_l2_noisy_student_475():
    model = torch.hub.load("rwightman/gen-efficientnet-pytorch",
                           "tf_efficientnet_l2_ns_475",
                           pretrained=True)
    model.classifier = nn.Identity()
    return model


@register_model()
def clip_laion():
    model = timm.create_model('vit_base_patch16_clip_224.laion2b_ft_in1k', pretrained=True)
    model = model.eval()
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    model.head = nn.Identity()
    return model, transforms

@register_model()
def random_rn50():
    model = resnet50()
    model.fc = nn.Identity()
    return model

@register_model()
def random_vit16l():
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.head = nn.Identity()
    return model

@register_model()
def cliplaion_vit32b():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
    model.forward = model.encode_image
    return model, preprocess

@register_model()
def cliplaion_cnxlxx():
    model, _, preprocess = open_clip.create_model_and_transforms('convnext_xxlarge', pretrained='laion2b_s34b_b82k_augreg')
    model.forward = model.encode_image
    return model, preprocess