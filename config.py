# -*- coding: utf-8 -*-
# @Time    : 2018/12/21 16:31
# @Author  : chenhao
# @FileName: config.py
# @Software: PyCharm

from easydict import EasyDict as edict

config = edict()

config.model = edict()
config.model.result_path = "./sample"
config.model.checkpoint_path = "./checkpoint"
config.model.log_path = "./log"
config.model.scale = 4
config.model.reblock_depth = 10
config.model.recursive_depth = 1

config.valid = edict()
config.valid.hr_folder_path = "./media/DATA/NTIRE2017/DIV2K_valid_HR"
config.valid.lr_folder_path = "./media/DATA/NTIRE2017/DIV2K_valid_LR_bicubic"

config.train = edict()
config.train.hr_folder_path = "./media/DATA/NTIRE2017/DIV2K_train_HR/"
config.train.lr_folder_path = "./media/DATA/NTIER2017/DIV2K_train_LR_bicubic"
config.train.batch_size = 4
config.train.in_patch_size = 64
config.train.out_patch_size = config.train.in_patch_size * config.model.scale
config.train.batch_size_each_folder = 30
config.train.log_write = False
config.train.lr_init = 5*1.e-6
config.train.lr_decay = 0.5
config.train.decay_iter = 10
config.train.betal = 0.9
config.train.n_epoch = 300
config.train.dump_intermediate_result = True

















