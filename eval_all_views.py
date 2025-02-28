import argparse
import os

from models import list_models
from sideviews import start_sideviews
from tools import str2bool

if __name__ == '__main__':
    def str2table(v):
        return v.split(',')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="resources/BabyVsModel/image_files/v0", type=str)
    parser.add_argument('--log_dir', default="logs", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--num_devices', default=1, type=int)
    parser.add_argument('--models', default=[], type=str2table)
    parser.add_argument('--start_models', default="", type=str)
    parser.add_argument('--whitebg', default=True, type=str2bool)

    args = parser.parse_args()
    args.dataset = "omnidataset"

    log_dir = os.path.join(args.log_dir, args.dataset+str(args.whitebg))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    models = list_models() if not args.models else args.models
    if args.start_models:
        models = models[models.index(args.start_models):]
    for model in models :
        print("Start", model)
        args.load = model
        start_sideviews(args, log_dir)