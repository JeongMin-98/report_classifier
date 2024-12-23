# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()

_C.DATA_DIR = ''
_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = 0
_C.WORKERS = 4
_C.PHASE = 'train'

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'dmis-lab/biobert-base-cased-v1.1-mnli'
_C.MODEL.EXTRA = None
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = 'dmis-lab/biobert-base-cased-v1.1-mnli'
_C.MODEL.NUM_LABELS = 6

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.JSON = ''
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'test'
_C.DATASET.DATA_FORMAT = 'csv'

# training data augmentation
# Add augmentation settings if required

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [10, 20]
_C.TRAIN.LR = 0.0001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.BATCH_SIZE_PER_GPU = 16
_C.TRAIN.NUM_EPOCHS = 10
_C.TRAIN.SHUFFLE = True

# LoRA specific settings
_C.LORA = CN()
_C.LORA.RANK = 8
_C.LORA.ALPHA = 16
_C.LORA.DROPOUT = 0.1
_C.LORA.TARGET_MODULES = ["query", "value"]

# testing
_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 16

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    # cfg.merge_from_list(args.opts)

    # if args.modelDir:
    #     cfg.OUTPUT_DIR = args.modelDir

    if args.log_dir:
        cfg.LOG_DIR = args.log_dir

    if args.data_dir:
        cfg.DATA_DIR = args.data_dir

    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
