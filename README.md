# BabyML


Code for our [ICLR 2025 blog post](https://iclr-blogposts.github.io/2025/blog/2025/toddlers-vs-vismodels/) : **Do vision models perceive objects like 
toddlers ?**

We will keep updating the repository with new models and novels experiments !

----

## News !


- **28/02/2025**: Add VideoMAE, V-JEPA models and other dinov2 models 
- **28/02/2025**: Upload the original code

----

## Install dependencies and datasets

`python3 -m pip install -r requirements.txt`


**Caricatures**: Images are available [there](https://osf.io/wbrd4/). To be extracted 
in `resources/`

#### Shape bias

From this [repo](https://github.com/alexatartaglini/developmental-shape-bias/tree/master/stimuli):

- Copy brodatz-textures to `resources/`
- Copy geirhos-masks to `resources/`
- Copy novel-masks to `resources`

`python3 create_datasets/create_dataset.py --name shape_simpletext`
`python3 create_datasets/create_dataset.py --name simpleshape_simpletext`


**OmniObject3D**  must be downloaded following this [repo](https://github.com/omniobject3d/OmniObject3D).
The code uses a hdf5 version of the dataset, constructed with:

`python3 create_datasets/create_hdf5_omni.py`

**Normal + Frakenstein silhouettes**: The dataset is provided on demand by [Prof. Baker](https://www.luc.edu/psychology/people/facultyandstaffdirectory/profiles/bakernicholas.shtml)

----


## Evaluation of models

The codes automatically download the models in the standard torch zoo directory. 
models.

For caricatures: 
```
python3 eval_all_caricatures.py --difficulty hard
python3 eval_all_caricatures.py --difficulty simple
```
For shape bias: 
```
python3 eval_all_shape_bias --dataset "shape_simpletext"
python3 eval_all_shape_bias --dataset "simpleshape_simpletext"
```
For view bias: `python3 eval_all_views.py`

For configural arrangement of parts: 

`python3 eval_conf_shape.py --difficulty hard`

----

## Citation 

If you use BabyML, please cite our blogpost:

```
@inproceedings{aubret2024perceive,
  author = {Aubret, Arthur and Triesch, Jochen},
  title = {Do vision models perceive objects like toddlers ?},
  booktitle = {ICLR Blog Track},
  year = {2025},
  date = {April 28, 2025},
  url  = {https://aubret.github.io/2025/blog/toddlers-vs-vismodels/}
}
```