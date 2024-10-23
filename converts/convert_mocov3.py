import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import config

args = config.parser.parse_args()


checkpoint = torch.load(args.load, map_location="cpu")
checkpoint = checkpoint["state_dict"]

new_state_dict = {}
for k, w in checkpoint.items():
    if "module.base_encoder" in k:
        k2 = ".".join(k.split(".")[2:])
        new_state_dict[k2] = w

dst_splits = args.load.split("/")
dst_splits[-1] = "converted_"+dst_splits[-1]
torch.save(new_state_dict, "/".join(dst_splits))
