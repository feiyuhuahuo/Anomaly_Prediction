import Dataset
from models.unet import UNet
import numpy as np
from utils import psnr_error
import os
import time
import torch
import argparse
from config import update_config
from sklearn import metrics
from Dataset import Label_loader
import matplotlib.pyplot as plt
import pdb

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--input_num', default='4', type=int, help='The frame number to be used to predict one frame.')
parser.add_argument('--color_type', default='colorful', type=str, help='The color type of the dataset.')
parser.add_argument('--trained_g', default='G_80000.pth', type=str, help='The pre-trained generator to evaluate.')
parser.add_argument('--show_curve', action='store_true',
                    help='Show the psnr curve real-timely, slightly influence fps.')
parser.add_argument('--show_heatmap', action='store_true',
                    help='Show the difference heatmap real-timely, slightly influence fps.')


def val(cfg, model=None):
    if model:
        generator = model
        generator.eval()
    else:
        generator = UNet(input_channels=12, output_channel=3).cuda().eval()
        generator.load_state_dict(torch.load('weights/' + cfg.trained_g)['net'])
        print(f'The pre-trained generator has been loaded with \'weights/{args.trained_g}\'.\n')

    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]

    fps = 0
    psnr_groups = []
    for i, folder in enumerate(video_folders):
        dataset = Dataset.test_dataset(folder, clip_length=5)
        plt.xlabel('frames')
        plt.ylabel('psnr')
        plt.xlim((0, len(dataset)))
        plt.ylim((20, 60))

        psnrs = []
        for j, clip in enumerate(dataset):
            input_frames = clip[0:12, :, :].unsqueeze(0).cuda()
            target_frame = clip[12:15, :, :].unsqueeze(0).cuda()

            G_frame = generator(input_frames)
            test_psnr = psnr_error(G_frame, target_frame).cpu().detach().numpy()
            psnrs.append(test_psnr)

            plt.plot(i, m, '-r')
            plt.pause(0.0001)

            torch.cuda.synchronize()
            end = time.time()
            if j > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                fps = 1 / (end - temp)
            temp = end
            print(f'\rDetecting: [{i:02d}] {j}/{len(dataset)}, {fps:.2f} fps.', end='')

        psnr_groups.append(psnrs)
    print('\nAll frames were detected, begin to compute AUC.')

    gt_loader = Label_loader(cfg, video_folders)
    gt = gt_loader()

    detected_num = len(psnr_groups)
    assert detected_num == len(gt), f'Ground truth has {len(gt)} videos, but got {detected_num} detected videos.'

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    for i in range(detected_num):
        distance = psnr_groups[i]
        distance -= min(distance)  # distance = (distance - min) / (max - min)
        distance /= max(distance)

        scores = np.concatenate((scores, distance), axis=0)
        labels = np.concatenate((labels, gt[i][4:]), axis=0)  # Exclude the first 4 unpredictable frames in gt.

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    print(f'AUC: {auc}')


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)
    # Uncomment this to test the AUC mechanism.
    # labels = [0,  0,   0,   0,   0,  1,   1,    1,   0,  1,   0,    0]
    # scores = [0, 1/8, 2/8, 1/8, 1/8, 3/8, 6/8, 7/8, 5/8, 8/8, 2/8, 1/8]
    # fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    # print(fpr)
    # print('~~~~~~~~~~~~`')
    # print(tpr)
    # print('~~~~~~~~~~~~`')
    # print(thresholds)
    # print('~~~~~~~~~~~~`')
