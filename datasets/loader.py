import json

from datasets import PascalVOCLoader
from datasets import CamvidLoader
from datasets import CityscapesLoader

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        'pascal': PascalVOCLoader,
        'camvid': CamvidLoader,
        'cityscapes': CityscapesLoader
    }[name]