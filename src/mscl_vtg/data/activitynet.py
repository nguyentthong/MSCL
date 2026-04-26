"""ActivityNet-Captions dataset adapter.  C3D video + GloVe text features."""
from .base import JsonAnnotationDataset


class ActivityNetDataset(JsonAnnotationDataset):
    """Uses the generic JSON annotation loader.

    Expected file: data/annotations/activitynet_{split}.json  OR  {split}.json
    """
    pass
