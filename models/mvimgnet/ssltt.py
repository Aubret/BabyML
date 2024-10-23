import re

from torch.utils import model_zoo
from torchvision.models import resnet50

model_urls = {
    'aasimclr': 'https://huggingface.co/aaubret/AASSL/resolve/main/aasimclr.pt',
    'simclr': 'https://huggingface.co/aaubret/AASSL/resolve/main/simclr.pt',
    'simclrtt': 'https://huggingface.co/aaubret/AASSL/resolve/main/simclrtt.pt',
    'cipersimclr': 'https://huggingface.co/aaubret/AASSL/resolve/main/cipersimclr.pt',
}


def mvimgnet(variant):
    model = resnet50()
    checkpoint = model_zoo.load_url(model_urls[variant])

    new_state_dict = {}
    for k, w in checkpoint.items():
        if re.search("^model.*", k):
            k = ".".join(k.split(".")[1:])
        if re.search("^projector.*", k):
            continue
        if re.search("^sup_lin*", k):
            continue
        if "action_head" in k and not "action_head" in args.keep_proj:
            continue
        # if "action_projector" in k and "action_projector" in args.keep_proj:
        #     new_k = ".".join(["head_action.layers"] + k.split(".")[2:])
        #     new_state_dict[new_k] = w
        # elif "equivariant_projector" in k and "equivariant_projector" in args.keep_proj:
        #     new_k = ".".join(["head_equivariant.layers"] + k.split(".")[2:])
        #     new_state_dict[new_k] = w
        # elif "equivariant_predictor" in k and "equivariant_predictor" in args.keep_proj:
        #     new_k = ".".join(["head_prediction.layers"] + k.split(".")[2:])
        #     new_state_dict[new_k] = w
        # else:
        new_state_dict[k] = w

    model.load_state_dict(new_state_dict)
    return model
