import argparse
import csv
import os

import lightning as L
import torch
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data import DataLoader

from datasets import DATASETS
from models import list_models
from models.registry import model_registry
from tools import load_model, get_features, get_transforms, str2bool


@torch.no_grad()
def sideviews(args):
    torch.set_float32_matmul_precision('medium')
    strategy = DDPStrategy(broadcast_buffers=False)
    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=strategy, precision="32-true")
    fabric.seed_everything(0)
    fabric.launch()

    # preprocess = get_transforms(args, norm=False)
    preprocess = get_transforms(args)
    if args.load in list_models():
        model = model_registry[args.load]()
        if isinstance(model, tuple) and len(model) > 1:
            model, preprocess = model
    else:
        model, preprocess = load_model(args, preprocess)

    dataset = DATASETS[args.dataset](args.data_root, subset_name="all", transform=preprocess, whitebg=args.whitebg)
    datasetside = DATASETS[args.dataset](args.data_root, subset_name="side", transform=preprocess, whitebg=args.whitebg)
    datasetfront = DATASETS[args.dataset](args.data_root, subset_name="front", transform=preprocess, whitebg=args.whitebg)
    datasetquarter = DATASETS[args.dataset](args.data_root, subset_name="quarter", transform=preprocess, whitebg=args.whitebg)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    dataloaderside = DataLoader(datasetside, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    dataloaderfront = DataLoader(datasetfront, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    dataloaderquarter = DataLoader(datasetquarter, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    dataloader, dataloaderside, dataloaderfront, dataloaderquarter = fabric.setup_dataloaders(dataloader, dataloaderside, dataloaderfront, dataloaderquarter)


    model = fabric.setup(model)
    model.eval()

    # for i, (img, _, _, _) in enumerate(dataloader):
    #     torchvision.utils.save_image(img, f"/home/fias/postdoc/gym_results/test_images/save_omni_side/{i}.png")
    #     if i > 2000: return

    features, cat, obj = get_features(dataloader, model, fabric)
    featuresside, catside, objside = get_features(dataloaderside, model, fabric)
    featuresfront, catfront, objfront = get_features(dataloaderfront, model, fabric)
    featuresquarter, catquarter, objquarter = get_features(dataloaderquarter, model, fabric)
    # catlist = cat.unique()
    objlist = obj.unique()

    objresa, catresa, allresa = [], [], []

    for o in objlist:
        mask_obj = o == obj
        mask_objs = o == objside
        mask_objf = o == objfront
        mask_objq = o == objquarter

        f = features[mask_obj]
        fs = featuresside[mask_objs]
        ff = featuresfront[mask_objf]
        fq = featuresquarter[mask_objq]

        c = cat[mask_obj][0].item()
        mask_cat = (cat == c) & (~mask_obj)
        mask_notcat = cat != c
        f_cat = features[mask_cat]
        if len(f_cat) == 0:
            f_cat = f


        objres, catres, allres = [], [], []
        for fsel in (fs, ff, fq):
            allobj = torch.nn.functional.cosine_similarity(fsel, f, dim=1)
            objres.append(allobj.mean())

            allcat = torch.nn.functional.cosine_similarity(fsel, f_cat, dim=1)
            catres.append(allcat.mean())

            all = 0
            cpt =0
            for f_all in features[mask_notcat].split(128):
                all += torch.nn.functional.cosine_similarity(fsel, f_all, dim=1).sum()
                cpt+=f_all.shape[0]
            allres.append(all/cpt)
        objresa.append(objres)
        catresa.append(catres)
        allresa.append(allres)

    objresa, catresa, allresa = torch.tensor(objresa), torch.tensor(catresa), torch.tensor(allresa)
    all_res = []
    for r in (objresa, catresa, allresa):

        all_res.append(((r[:, 0] > r[:, 1]) & (r[:, 0] > r[:, 2])).float().mean().cpu().item())
        all_res.append(((r[:, 1] > r[:, 0]) & (r[:, 1] > r[:, 2])).float().mean().cpu().item())
        all_res.append(((r[:, 2] > r[:, 1]) & (r[:, 2] > r[:, 0])).float().mean() .cpu().item())

        all_res.append(((r[:, 0] < r[:, 1]) & (r[:, 0] < r[:, 2])).float().mean().cpu().item())
        all_res.append(((r[:, 1] < r[:, 0]) & (r[:, 1] < r[:, 2])).float().mean().cpu().item())
        all_res.append(((r[:, 2] < r[:, 1]) & (r[:, 2] < r[:, 0])).float().mean() .cpu().item())



    return all_res

def start_sideviews(args, log_dir):
    headers = []
    name_test = args.load.split('/')[-1].split(".")[0] if not args.load in list_models() else args.load
    for r in ["obj", "cat", "all"]:
        for types in ("max", "min"):
            headers.append(r+"_side_"+types)
            headers.append(r+"_front_"+types)
            headers.append(r+"_quarter_"+types)


    with open(os.path.join(log_dir, f"{name_test}_{args.dataset}_views_bg{args.whitebg}.csv"), "w") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(headers)
        wcsv.writerow(sideviews(args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="resources/OmniDataset/", type=str)
    parser.add_argument('--log_dir', default="logs", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--num_devices', default=1, type=int)
    parser.add_argument('--load', default="random", type=str)
    parser.add_argument('--whitebg', default=True, type=str2bool)
    args = parser.parse_args()
    args.dataset = "omnidataset"

    args.log_dir = os.path.join(args.log_dir, args.dataset)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    start_sideviews(args, args.log_dir)