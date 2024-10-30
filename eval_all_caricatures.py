import argparse
import os, sys

from caricatures import start_caricatures
from models import list_models

list_models_rn50 = ['clip_rn50', 'byol_rn50', 'r3m', 'vip', 'aasimclr','simclrtt', 'mocov2', 'bagnet33', 'resnet50_l2_eps1', "bagnet17"]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="../datasets/BabyVsModel/image_files/v0", type=str)
    parser.add_argument('--log_dir', default="logs", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--num_devices', default=1, type=int)
    parser.add_argument('--subset', default="full", type=str)
    args = parser.parse_args()
    args.dataset = "babymodel"

    log_dir = os.path.join(args.log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for model in list_models():
    # for model in list_models_rn50:
        print("Start", model)
        args.load = model
        start_caricatures(args, log_dir, args.subset)