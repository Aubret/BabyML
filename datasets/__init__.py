from datasets.babymodel import IndividualObjectsDataset
from datasets.frankenstein import FrankensteinDataset
from datasets.shepardmetzler import ShepardMetzler

DATASETS = {
    "babymodel": IndividualObjectsDataset,
    "frankenstein": FrankensteinDataset,
    "shepardmetzler": ShepardMetzler
}