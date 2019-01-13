# -*- coding: utf-8 -*-
# @Time    : 2018/12/21 16:31
# @Author  : chenhao
# @FileName: config.py
# @Software: PyCharm

from easydict import EasyDict as edict
import json

config = edict()

config.model = edict()
config.model.result_path = "./sample/"
config.model.checkpoint_path = "./checkpoint/"
config.model.log_path = "./log/"
config.model.scale = 4
config.model.reblock_depth = 10
config.model.recursive_depth = 1

config.valid = edict()
config.valid.hr_folder_path = "./media/DATA/NTIRE2017/DIV2K_valid_HR/"
config.valid.lr_folder_path = "./media/DATA/NTIRE2017/DIV2K_valid_LR_bicubic/X4/"

config.train = edict()
config.train.hr_folder_path = "./media/DATA/NTIRE2017/DIV2K_train_HR/"
config.train.lr_folder_path = "./media/DATA/NTIRE2017/DIV2K_train_LR_bicubic/X4/"
config.train.batch_size = 4
config.train.in_patch_size = 64
config.train.out_patch_size = config.train.in_patch_size * config.model.scale
config.train.batch_size_each_folder = 30
config.train.log_write = False
config.train.lr_init = 1.e-5
config.train.lr_decay = 0.5
config.train.decay_iter = 50
config.train.betal = 0.9
config.train.n_epoch = 150
config.train.dump_intermediate_result = True
config.train.train_path = "./sample/train/"

config.test = edict()
config.test.test_Data = "./media/DATA/TEST/"
config.test.test_path = "./sample/test/"

# Convert Data to Json
def log_config(filename, cfg):
    with open(filename, "w") as f:
        f.write(json.dumps(cfg, indent=4))
