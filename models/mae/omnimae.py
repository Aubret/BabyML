from models.mae.models.omnivore_mae import vit_large_mae_pretraining, vit_base_mae_pretraining


def OmniMAE(backbone, **kwargs):
    if backbone == "vitb":
        model = vit_base_mae_pretraining(pretrained=True, **kwargs)
    else:
        model = vit_large_mae_pretraining(pretrained=True, **kwargs)
    return model
