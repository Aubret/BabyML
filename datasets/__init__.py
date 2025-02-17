from datasets.babymodel import IndividualObjectsDataset
from datasets.frankenstein import FrankensteinDataset
from datasets.omnidataset import OmniDataset
from datasets.shepardmetzler import ShepardMetzler
from datasets.triplet import TripletDataset

DATASETS = {
    "omnidataset": OmniDataset,
    "babymodel": IndividualObjectsDataset,
    "frankenstein": FrankensteinDataset,
    "shepardmetzler": ShepardMetzler,
    "img_img_shapetext": TripletDataset,
    "shape_simpletext": TripletDataset,
    "simpleshape_simpletext": TripletDataset,
}