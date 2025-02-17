import argparse
import os, sys
# sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/..")
from configural_shape import start_configural_shape
from models import list_models
from shape_bias import start_shape_bias
from tools import str2bool

list_models_rn50 = ['clip_rn50', 'byol_rn50', 'r3m', 'vip', 'aasimclr','simclrtt', 'mocov2', 'bagnet33', 'resnet50_l2_eps1', "bagnet17"]
#list_models_rn50 = ['clip_vit', "mae_vitl16","dinov2", 'byol_rn50', 'r3m', 'vip', 'aasimclr','simclrtt', 'vc1', 'bagnet33', 'resnet50_l2_eps0_25', 'cv1']



if __name__ == '__main__':
    def str2table(v):
        return v.split(',')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="../datasets/shapebias/", type=str)
    parser.add_argument('--dataset', default="img_img_shapetext", type=str, choices=["img_img_shapetext","shape_simpletext","simpleshape_simpletext"])
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