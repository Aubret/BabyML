from timm.models.resnet import resnet200
from torch.utils import model_zoo
from torchvision.models import resnet50

model_urls = {
    'rn50': 'https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl',
    # 'rn200x4': 'https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res200x2.pkl',
}


def Byol(backbone):
    model = resnet50()
    model.load_state_dict(model_zoo.load_url(model_urls[backbone]))
    return model
