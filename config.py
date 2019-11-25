#!/usr/bin/env python 
# -*- coding:utf-8 -*-

default_config = {'dataset': 'avenue',
                  'train_data': '/home/feiyu/Data/avenue/training/frames',
                  'test_data': '/home/feiyu/Data/avenue/testing/frames',
                  'batch_size': 4,
                  'g_lr': 0.0002,
                  'd_lr': 0.00002,
                  'num_input': 4,
                  'iters': 80000,
                  }


class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + 'Config' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def update_config(args=None):
    default_config['batch_size'] = args.batch_size
    default_config['dataset'] = args.dataset
    default_config['train_data'] = f'/home/feiyu/Data/{args.dataset}/training/frames'
    default_config['test_data'] = f'/home/feiyu/Data/{args.dataset}/testing/frames'

    assert args.dataset_type in ['color', 'grey'], 'Dataset type can only be color scale or grey scale.'
    if args.dataset_type == 'grey':
        default_config['g_lr'] = 0.0001
        default_config['d_lr'] = 0.00001

    return dict2class(default_config)  # change dict contents to class attributes
