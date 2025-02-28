from functools import partial

from timm.models import VisionTransformer
from torch import nn
from torch.utils import model_zoo


# model_urls = {
#     "vc1": "https://dl.fbaipublicfiles.com/eai-vc/vc1_vitl.pth"
# }

def VC1_Model():
    model = vit_large_patch16()
    # model.load_state_dict(model_zoo.load_url(model_urls["vc1"]))
    state_dict = model_zoo.load_url("https://dl.fbaipublicfiles.com/eai-vc/vc1_vitl.pth")
    state_dict = state_dict["model"]
    model.head = nn.Identity()
    del state_dict["mask_token"]
    model.load_state_dict(state_dict)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
