#!/usr/bin/env python 
# -*- coding:utf-8 -*-

train_data = '/home/feiyu/Data/avenue/training/frames'
test_data = '/home/feiyu/Data/avenue/testing/frames'

generator_model = '../pth_model/ano_pred_avenue_generator_2.pth'
discriminator_model = '../pth_model/ano_pred_avenue_discriminator_2.pth'
liteflow_model = '../liteFlownet/network-default.pytorch'

writer_path = '../log/ano_pred_avenue'
flownet2SD_model_path = 'flownet2/FlowNet2-SD.pth'

batch_size = 2
epochs = 20000
pretrain = False

# color dataset
g_lr = 0.0002
d_lr = 0.00002

num_clips = 5
num_his = 1
num_unet_layers = 4

num_channels = 3  # avenue is 3, UCSD is 1
discriminator_channels = [128, 256, 512, 512]
