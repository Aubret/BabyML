from functools import partial

from timm.models import VisionTransformer
from torch import nn
from torch.utils import model_zoo

model_urls = {
    'vitb': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
    'vitl': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
}


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


def MAE(backbone):
    model = vit_base_patch16() if backbone == "vitb" else vit_large_patch16()
    model.load_state_dict(model_zoo.load_url(model_urls[backbone]))
    return model
