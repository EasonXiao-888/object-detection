# Copyright (c) OpenMMLab. All rights reserved.
from .coco import CocoDataset
from .builder import DATASETS
import numpy as np

@DATASETS.register_module()
class dota1_5(CocoDataset):

    CLASSES = ('plane', 'baseball-diamond',
                'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle',
                'ship', 'tennis-court',
                'basketball-court', 'storage-tank',
                'soccer-ball-field', 'roundabout',
                'harbor', 'swimming-pool',
                'helicopter', 'container-crane')

 
