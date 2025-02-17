import argparse
import os, sys

from models import list_models
from scripts.invariance import start_invariance

list_models_rn50 = ['clip_rn50', 'byol_rn50', 'r3m', 'vip', 'aasimclr','simclrtt', 'mocov2', 'bagnet33', 'resnet50_l2_eps1', "bagnet17"]



if __name__ == '__main__':
    def str2table(v):
        return v.split(',')


    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
            return False
        else:
            raise Exception('Boolean value expected.')


    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data_root', default="../datasets/ShepardMetzler/", type=str)
    parser.add_argument('--log_dir', default="logs", type=str)
    parser.add_argument('--load', default="random", type=str)
    parser.add_argument('--model', default="resnet50", type=str)
    parser.add_argument('--head', default="", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--load_strict', default=True, type=str2bool)
    parser.add_argument('--rotation', default="", type=str)
    parser.add_argument('--keep_proj', default=["projector"], type=str2table)
    parser.add_argument('--models', default=[], type=str2table)
    parser.add_argument('--dense_features', default=False, type=str2bool)

    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--num_devices', default=1, type=int)

    args = parser.parse_args()
    args.dataset = "shepardmetzler"
    args.pos_subset = "rotated"
    args.neg_subset = "mirror"
    # assert args.head == "action_prediction", "Need action prediction module"
    args.log_dir = os.path.join(args.log_dir, args.dataset, "inv")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    models = list_models() if not args.models else args.models
    for model in models :
    # for model in list_models_rn50:
        print("Start", model)
        args.load = model
        start_invariance(args, args.log_dir)