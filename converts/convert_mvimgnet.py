import os
import re

import torch
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import config

parser = config.parser
parser.add_argument('--keep_proj', default=[], type=config.str2table)
args = config.parser.parse_args()


checkpoint = torch.load(args.load, map_location="cpu")
checkpoint = checkpoint["model"]

new_state_dict = {}
for k, w in checkpoint.items():
    if re.search("^model.*", k):
        k = ".".join(k.split(".")[1:])
    if re.search("^projector.*", k):
        continue
    if re.search("^sup_lin*", k):
        continue
    # if "action_projector" in k and not "action_projector" in args.keep_proj:
    #     continue
    if "action_head" in k and not "action_head" in args.keep_proj:
        continue

    if "action_projector" in k and "action_projector" in args.keep_proj:
        new_k = ".".join(["head_action.layers"] + k.split(".")[2:])
        new_state_dict[new_k] = w
    elif "equivariant_projector" in k and "equivariant_projector" in args.keep_proj:
        new_k = ".".join(["head_equivariant.layers"] + k.split(".")[2:])
        new_state_dict[new_k] = w
    elif "equivariant_predictor" in k and "equivariant_predictor" in args.keep_proj:
        new_k = ".".join(["head_prediction.layers"] + k.split(".")[2:])
        new_state_dict[new_k] = w
    else:
        new_state_dict[k] = w
dst_splits = args.load.split("/")
dst_splits[-1] = "converted_"+dst_splits[-1]
torch.save(new_state_dict, "/".join(dst_splits))
