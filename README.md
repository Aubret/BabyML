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
python3 scripts/test_mae_seq.py --load videomae
```


```
python3 scripts/action_prediction_objects.py --load ../gym_results/imgnet/imgnet57_actequirav2_det+/03-12-24_11-12_imgnet57_actequirav2_det+_1/models/epoch_30.pt
python3 scripts/action_prediction_objects_omni.py --model resnet18 --load ../gym_results/shapenet/omni11_sslact/16-07-24_11-52_omni11_sslact_0/models/epoch_100.pt
```