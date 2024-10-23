import torch
import clip
import torchvision.models
from torchvision.models import ViT_L_16_Weights

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

def get_transforms(dataset):
    mean, std, image_size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 512
    # mean, std, image_size = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 224
    if dataset in ["frankenstein", "shepardmetzler"]:
        #Silhouettes are close to the border of the image
        return trv2.Compose([trv2.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC), trv2.ToImage(),
                             trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])
    return trv2.Compose([trv2.Resize(image_size, interpolation=InterpolationMode.BICUBIC), trv2.CenterCrop(image_size),
                         trv2.ToImage(), trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])


def get_features(dataloader, model, fabric):
    features, labels, img_ids = [], [], []
    for img, label, img_id in dataloader:
        # mean, std, image_size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
        # plt.imshow((img[0].permute(1, 2, 0) * torch.tensor(std, device=img.device) + torch.tensor(mean, device=img.device)).cpu().numpy())
        # plt.title(f"{DATASETS['frankenstein'].label_to_class[int(label[0])]}")
        # plt.show()
        f = model(img)
        f = model.head(f)
        features.append(f)
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
