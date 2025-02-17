from torch import nn
from torch.utils import model_zoo

from models.jepa.transformer import vit_huge, vit_large

model_urls = {
    'vitl': 'https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar',
    'vith': 'https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar',
}

def VJEPA(backbone, n_frame=2):
    model = vit_large(num_frames=n_frame) if backbone == "vitl" else vit_huge(num_frames=n_frame)
    state_dict = model_zoo.load_url(model_urls[backbone], map_location="cpu")
    state_dict = state_dict["encoder"]
    for k in list(state_dict.keys()):
        if not "pos_embed" in k:
            new_k = ".".join(k.split(".")[2:])
            state_dict[new_k] = state_dict[k]
        del state_dict[k]
    model.head = nn.Identity()
    model.load_state_dict(state_dict, strict=False)
    return model
