import math

import torch
import clip
import torchvision.models
from torchvision.models import ViT_L_16_Weights
from tqdm import tqdm

from models import resnet18_default, resnet50_default
from torchvision.transforms import v2 as trv2, InterpolationMode

from models.heads import ActionMultiLayerProj, MultiLayerProj
from models.vit import vit_large_patch16, vit_large_patch14

BACKBONES = {
    "resnet18": resnet18_default,
    "resnet50": resnet50_default,
    "ViT-L/16": vit_large_patch16,
    "ViT-L/14": vit_large_patch14
}


def module_finding(m, pattern):
    try:
        for c, cm in m.named_children():
            # if not hasattr(c, "__name__"): return module_finding(c, pattern)
            if pattern in c:
                return c
            child = module_finding(cm, pattern)
            if child:
                return c + "." + child
        return False
    except:
        return False

def module_finding_class(m, pattern):
    if pattern in m.__class__.__name__:
        return True
    for c, cm in m.named_children():
        child = module_finding_class(cm, pattern)
        if child:
            return child
    return False



def hook_dense_features(model):
    def patch_hook_resnet(modeld):
        def fn(module, __, output):
            modeld.dense_features = output
            return output
        return fn
    def patch_hook_vit(modeld):
        def fn(module, __, output):
            pf=output
            if output.shape[1] in [16, 32, 1, 3]:
                pf = torch.permute(pf, (1, 0, 2 ))
            if hasattr(modeld, "num_register_tokens"):
                #Dino Vision transformer special case with several "register tokens"
                pf = pf[:, modeld.num_register_tokens+1:]
            elif pf.shape[1]%2 == 1:
                pf = pf[:, 1:]
            pf = torch.permute(pf, (0, 2, 1 ))
            # module.dense_features = pf.view(pf.shape[0], -1, int(math.sqrt(pf.shape[2])),int(math.sqrt(pf.shape[2])))
            modeld.dense_features = pf.view(pf.shape[0], -1, int(math.sqrt(pf.shape[2])),int(math.sqrt(pf.shape[2])))
            return output
        return fn

    # print({k:v for k,v in model.named_modules()})
    # print(model.module.extra_repr())
    # print(module_finding(model.module, "layer4"))

    module = model.module if hasattr(model, "module") else model
    if module_finding_class(module, "ResNet") or module_finding_class(module, "BagNet"):
        repr_layer = module_finding(module, "layer4")
        model.get_submodule(repr_layer).register_forward_hook(patch_hook_resnet(model))
    elif module_finding_class(module, "ConvNeXt"):
        repr_layer = module_finding(module, "stages")
        model.get_submodule(repr_layer).register_forward_hook(patch_hook_resnet(model))
    elif module_finding_class(module, "EfficientNet"):
        repr_layer = module_finding(module, "blocks")
        model.get_submodule(repr_layer).register_forward_hook(patch_hook_resnet(model))

    elif module_finding_class(module, "VisionTransformer"):
        repr_layer = module_finding(module, "blocks")
        if not repr_layer:
            repr_layer = module_finding(module, "layers")

        for add_name in [".23", ".11", ".encoder_layer_23", ".encoder_layer_11"]:
            try:
                if model.get_submodule(repr_layer+add_name):
                    repr_layer = repr_layer+add_name
                    break
            except:
                pass
        model.get_submodule(repr_layer).register_forward_hook(patch_hook_vit(model))
    else:
        print(module)
        raise Exception(f"Dense features not available for this model {model.__class__.__name__}")
    return model


def load_model(args, preprocess=None):


    ### Remote checkpoints
    if args.load == "clip":
        model, preprocess = clip.load("ViT-L/14", device="cpu")
        model.forward = model.encode_image
        add_head(model, args.head)
        return model, preprocess

    if args.load == "dinov2":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        add_head(model, args.head)
        return model, preprocess

    if args.load == "ViT-L/16":
        model = torchvision.models.vit_l_16(weights = ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        add_head(model, args.head)
        return model, preprocess


    ### Local checkpoints
    model = BACKBONES[args.model]()
    add_head(model, args.head)

    if args.load != "random":
        checkpoint = torch.load(args.load, map_location="cpu")
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        to_remove = []
        for k in checkpoint.keys():
            if k.startswith("fc."):
                to_remove.append(k)
            if k.startswith("classifier."):
                to_remove.append(k)

        for k in to_remove:
            del checkpoint[k]
        model.load_state_dict(checkpoint, strict=args.load_strict)
    return model, preprocess




def get_transforms(args, norm=True):
    # mean, std, image_size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 512
    if "omni" in args.load or not norm:
        mean, std, image_size = (0, 0, 0), (1, 1, 1), 224
    else:
        mean, std, image_size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
        # mean, std, image_size = (0, 0, 0), (1, 1, 1), 224

        # mean, std, image_size = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 224
    if args.dataset == "shepardmetzler":
        # mean, std, image_size = (0, 0, 0), (1, 1, 1), 224
        # mean, std, image_size = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 224
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
            f = model.dense_features.flatten(1)
        features.append(torch.clone(f))
        labels.append(label)
        img_ids.append(img_id)

    features, labels, img_ids = torch.cat(features, dim=0), torch.cat(labels, dim=0), torch.cat(img_ids, dim=0)
    data = fabric.all_gather((features, labels, img_ids))
    data = (d.flatten(0,1).squeeze() for d in data)
    return data

def add_head(model, head):
    if head == "action_predictor":
        model.add_module("head", ActionMultiLayerProj(2, 2*2048, 4096, 5, bias=False))
    elif head == "action_prediction":
        model.head = ActionMultiLayerProj(2, 2*2048, 2048, 128, bias=False)
        model.head_equivariant = MultiLayerProj(1, 2048, 2048, 128, bias=False)
        model.head_prediction = MultiLayerProj(2, 128, 2048, 128, bias=False)
    else:
        model.head = torch.nn.Identity()

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
