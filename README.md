# BabyML





---
### Models

BYOL downloaded there: 
https://github.com/google-deepmind/deepmind-research/tree/master/byol

Mocov3 downloaded there:
https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md

Dino downloaded there:
https://github.com/facebookresearch/dino

MVImgNet models:
https://huggingface.co/aaubret/AASSL/tree/main

Clip: script download

Dinov2: script download



### Examples

```
python3 ooo_category.py --dataset frankenstein --load ../models/imgnet_100ep/converted_aabyol.ckpt --data_root ../datasets/baker/ --subset full
python3 odd_one_out.py --dataset babymodel --load ../models/imgnet_100ep/converted_aabyol.ckpt --data_root ../datasets/BabyVsModel/image_files/v0 --subset full
python3 ooo_subset.py --dataset babymodel --load ../models/imgnet_100ep/converted_aabyol.ckpt --data_root ../datasets/BabyVsModel/image_files/v0 --subset full
python3 posout_negin.py --dataset babymodel --load clip --data_root ../datasets/BabyVsModel/image_files/v0 --subset full --batch_size 8 --device cpu --subset geons
```

#### Mental rotations

```
python3 converts/convert_mvimgnet.py --load ../models/mvimgnet/sslactequi/sslactequiradet.pt --keep_proj action_projector,equivariant_projector,equivariant_predictor
python3 scripts/mental_rotation.py --load ../models/mvimgnet/sslactequi/converted_sslactequira.pt --data_root ../datasets/ShepardMetzler/ --dataset shepardmetzler
```
