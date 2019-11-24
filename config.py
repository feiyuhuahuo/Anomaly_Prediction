#!/usr/bin/env python 
# -*- coding:utf-8 -*-

config = {'train_data': '/home/feiyu/Data/avenue/training/frames',
          'test_data': '/home/feiyu/Data/avenue/testing/frames',
          'batch_size': 4,
          'g_lr': 0.0002,
          'd_lr': 0.00002,
          'num_input': 4,
          }


class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        for k, v in vars(self).items():
            print(f'{k}: {v}')


def update_config(args):
    config['batch_size'] = args.batch_size

    assert args.dataset_type in ['colorful', 'grayscale'], 'Dataset type can only be colorful or greyscale.'
    if args.dataset_type == 'colorful':
        pass
    else:
        config['g_lr'] = 0.0001
        config['d_lr'] = 0.00001

    return dict2class(config)
