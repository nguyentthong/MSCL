"""Charades-STA dataset adapter.  I3D video + GloVe text features."""
from .base import JsonAnnotationDataset


class CharadesDataset(JsonAnnotationDataset):
    """Uses the generic JSON annotation loader.

    Expected file: data/annotations/charades_{split}.json  OR  {split}.json
    """
    pass
