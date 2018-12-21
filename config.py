# -*- coding: utf-8 -*-
# @Time    : 2018/12/21 16:31
# @Author  : chenhao
# @FileName: config
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
