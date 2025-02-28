import argparse
import os

from caricatures import start_caricatures
from models import list_models

if __name__ == '__main__':
    def str2table(v):
        return v.split(',')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="resources/BabyVsModel/image_files/v0", type=str)
    parser.add_argument('--log_dir', default="logs", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--num_devices', default=1, type=int)
    parser.add_argument('--subset', default="full", type=str)
    parser.add_argument('--models', default=[], type=str2table)
    parser.add_argument('--difficulty', default="simple", type=str)
    args = parser.parse_args()
    args.dataset = "babymodel"

    log_dir = os.path.join(args.log_dir, args.dataset+("hard" if args.difficulty == "hard" else ""))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    models = list_models() if not args.models else args.models
    for model in models :
        print("Start", model)
        args.load = model
        start_caricatures(args, log_dir, args.subset)