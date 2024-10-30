#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import argparse
import datetime

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
parser.add_argument('--data_root', default="data", type=str)
parser.add_argument('--log_dir', default="logs", type=str)
parser.add_argument('--load', default="random", type=str)
parser.add_argument('--model', default="resnet50", type=str)
parser.add_argument('--head', default="", type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--load_strict', default=True, type=str2bool)

parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--num_devices', default=1, type=int)

