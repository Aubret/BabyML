import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import config
import re
args = config.parser.parse_args()


checkpoint = torch.load(args.load, map_location="cpu")["state_dict"]

new_state_dict = {}
for k, w in checkpoint.items():
    if "projector" in k:
        continue
    if "head" in k:
        continue
    if "momentum" in k:
        continue
    if "action_bn" in k:
        print(k, w)
        continue
    if re.search("^predictor.*", k):
        continue

    if "classifier" in k:
        new_k = k
    elif "action_predictor" in k:
        new_k = ".".join(["head"] + k.split(".")[1:])
    else:
        new_k = ".".join(k.split(".")[1:])

    new_state_dict[new_k] = w
dst_splits = args.load.split("/")
dst_splits[-1] = "converted_"+dst_splits[-1]
torch.save(new_state_dict, "/".join(dst_splits))
