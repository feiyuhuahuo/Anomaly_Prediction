#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import glob

share_config = {'mode': 'training',
                'dataset': 'avenue',
                'img_size': (256, 256),
                'data_root': '/home/feiyu/Data/',  # remember the final '/'
                'color_type': 'colorful',
                'input_num': 4}


class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + f'{self.mode} cfg' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def update_config(args=None, mode=None):
    share_config['mode'] = mode
    share_config['dataset'] = args.dataset
    share_config['input_num'] = args.input_num
    share_config['img_size'] = args.img_size

    assert args.color_type in ['colorful', 'grey'], 'Color type can only be \'colorful\' or \'grey\'.'
    share_config['color_type'] = args.color_type

    if mode == 'train':
        share_config['batch_size'] = args.batch_size
        share_config['train_data'] = share_config['data_root'] + args.dataset + '/training/frames'
        share_config['test_data'] = share_config['data_root'] + args.dataset + '/testing/frames'
        share_config['g_lr'] = 0.0002 if args.color_type == 'colorful' else 0.0001
        share_config['d_lr'] = 0.00002 if args.color_type == 'colorful' else 0.00001
        share_config['resume'] = glob.glob(f'weights/{args.resume}*')[0] if args.resume else None
        share_config['iters'] = args.iters
        share_config['show_flow'] = args.show_flow
        share_config['save_interval'] = args.save_interval
        share_config['val_interval'] = args.val_interval
        share_config['flownet'] = args.flownet

    elif mode == 'test':
        share_config['test_data'] = share_config['data_root'] + args.dataset + '/testing/frames'
        share_config['trained_g'] = args.trained_g
        share_config['show_curve'] = args.show_curve
        share_config['show_heatmap'] = args.show_heatmap

    return dict2class(share_config)  # change dict keys to class attributes
