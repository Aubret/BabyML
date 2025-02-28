import math

import torch
from torchvision.transforms import v2 as trv2, InterpolationMode
from tqdm import tqdm

from models import resnet18_default, resnet50_default
from models.vit import vit_large_patch16, vit_large_patch14

BACKBONES = {
    "resnet18": resnet18_default,
    "resnet50": resnet50_default,
    "ViT-L/16": vit_large_patch16,
    "ViT-L/14": vit_large_patch14
}


def hook_dense_features(model):
    def patch_hook_resnet():
        def fn(module, __, output):
            module.dense_features = output
    def patch_hook_vit():
        def fn(module, __, output):
            pf = output.squeeze()[:, 1:]  # No token
            pf = torch.permute(pf, (0, 2, 1))
            module.dense_features = pf.view(pf.shape[0], -1, int(math.sqrt(pf.shape[2])),int(math.sqrt(pf.shape[2])))

        return fn
    if "Resnet" in model.__class__.__name__:
        layer = "pool"
        model.get_submodule(layer).register_forward_hook(patch_hook_resnet(model))

    elif "EfficientNet" in model.__class__.__name__:
        layer = "pool"
        model.get_submodule(layer).register_forward_hook(patch_hook_resnet(model))

    elif "Vision" in model.__class__.__name__:
        layer = "block.11"
        model.get_submodule(layer).register_forward_hook(patch_hook_vit(model))
    else:
        raise Exception("Dense features not available for this model")



def load_model(args, preprocess=None):
    ### Local checkpoints
    model = BACKBONES[args.model]()
    return model, preprocess




def get_transforms(args, norm=True):
    if "omni" in args.load or not norm:
        mean, std, image_size = (0, 0, 0), (1, 1, 1), 224
    else:
        mean, std, image_size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224

    if args.dataset == "shepardmetzler":
        return trv2.Compose(
            [trv2.Resize(image_size, interpolation=InterpolationMode.BICUBIC), trv2.CenterCrop(image_size),
             trv2.ToImage(), trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])

    if args.dataset in ["frankenstein", "shepardmetzler"]:
        #Silhouettes are close to the border of the image
        return trv2.Compose([trv2.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC), trv2.ToImage(),
                             trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])
    return trv2.Compose([trv2.Resize(image_size, interpolation=InterpolationMode.BICUBIC), trv2.CenterCrop(image_size),
                         trv2.ToImage(), trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])


@torch.no_grad()
def get_features(dataloader, model, fabric, dense_features=False):
    features, labels, img_ids = [], [], []
    for r in tqdm(dataloader):
        img, label, img_id = r[0], r[1], r[2]
        f = model(img)
        if dense_features:
            f = model.dense_features
        features.append(torch.clone(f))
        labels.append(label)
        img_ids.append(img_id)

    features, labels, img_ids = torch.cat(features, dim=0), torch.cat(labels, dim=0), torch.cat(img_ids, dim=0)
    data = fabric.all_gather((features, labels, img_ids))
    data = (d.flatten(0,1).squeeze() for d in data)
    return data

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
