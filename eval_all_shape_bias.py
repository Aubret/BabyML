import argparse
import os

# sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/..")
from models import list_models
from shape_bias import start_shape_bias
from tools import str2bool

if __name__ == '__main__':
    def str2table(v):
        return v.split(',')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="resources/shapebias/", type=str)
    parser.add_argument('--dataset', default="shape_simpletext", type=str, choices=["shape_simpletext","simpleshape_simpletext"])
    parser.add_argument('--log_dir', default="logs", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--num_devices', default=1, type=int)
    parser.add_argument('--subset', default="full", type=str)
    parser.add_argument('--whitebg', default=True, type=str2bool)
    parser.add_argument('--models', default=[], type=str2table)


    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    models = list_models() if not args.models else args.models
    for model in models:
        print("Start", model)
        args.load = model
        start_shape_bias(args, log_dir)