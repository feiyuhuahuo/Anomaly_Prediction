import numpy as np
import os
import time
import torch
import argparse
import cv2
from sklearn import metrics
import matplotlib.pyplot as plt

from config import update_config
from Dataset import Label_loader
from utils import psnr_error
import Dataset
from models.unet import UNet

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--input_num', default='4', type=int, help='The frame number to be used to predict one frame.')
parser.add_argument('--img_size', default=(256, 256), type=tuple, help='The image size for training and evaluating.')
parser.add_argument('--color_type', default='colorful', type=str, help='The color type of the dataset.')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--show_curve', action='store_true', help='Show the psnr curve real-timely, this drops fps.')
parser.add_argument('--show_heatmap', action='store_true',
                    help='Show the difference heatmap real-timely, this drops fps.')


def val(cfg, model=None):
    if model:  # This is for testing during training.
        generator = model
        generator.eval()
    else:
        generator = UNet(input_channels=12, output_channel=3).cuda().eval()
        generator.load_state_dict(torch.load('weights/' + cfg.trained_model)['net_g'])
        print(f'The pre-trained generator has been loaded from \'weights/{cfg.trained_model}\'.\n')

    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]

    fps = 0
    psnr_group = []

    if not model and cfg.show_curve:
        fig = plt.get_current_fig_manager()
        fig.window.setGeometry(550, 200, 600, 400)  # This works for QT backend, for other backends, check this ⬃⬃⬃.
        # https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
        plt.xlabel('frames')
        plt.ylabel('psnr')
        plt.grid(ls='--')

        cv2.namedWindow('target frames', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('target frames', 384, 384)
        cv2.moveWindow("target frames", 100, 100)

    if not model and cfg.show_heatmap:
        cv2.namedWindow('difference map', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('difference map', 384, 384)
        cv2.moveWindow('difference map', 100, 550)

    for i, folder in enumerate(video_folders):
        dataset = Dataset.test_dataset(cfg, folder)

        if not model and cfg.show_curve:
            js = []
            plt.clf()
            ax = plt.axes(xlim=(0, len(dataset)), ylim=(0, 50))
            ax.set_title('difference heatmap')
            line, = ax.plot([], [], '-b')

        psnrs = []
        for j, clip in enumerate(dataset):
            input_np = clip[0:12, :, :]
            target_np = clip[12:15, :, :]
            input_frames = torch.from_numpy(input_np).unsqueeze(0).cuda()
            target_frame = torch.from_numpy(target_np).unsqueeze(0).cuda()

            G_frame = generator(input_frames)
            test_psnr = psnr_error(G_frame, target_frame).cpu().detach().numpy()
            psnrs.append(float(test_psnr))

            if not model and cfg.show_curve:
                cv2_frame = ((target_np + 1) * 127.5).transpose(1, 2, 0).astype('uint8')
                cv2.imshow('target frames', cv2_frame)
                cv2.waitKey(1)

                js.append(j)
                line.set_xdata(js)  # This keeps the existing figure and just updates the X-axis data and Y-axis data,
                line.set_ydata(psnrs)  # which is faster, but still not perfect. Maybe multi-thread could be concerned.
                plt.pause(0.001)

            if not model and cfg.show_heatmap:
                diff_map = torch.sum(torch.abs(G_frame - target_frame).squeeze(), 0)
                diff_map -= diff_map.min()  # Normalize to 0 ~ 255.
                diff_map /= diff_map.max()
                diff_map *= 255
                diff_map = diff_map.cpu().detach().numpy().astype('uint8')

                heat_map = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
                cv2.imshow('difference map', heat_map)
                cv2.waitKey(1)

            torch.cuda.synchronize()
            end = time.time()
            if j > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                fps = 1 / (end - temp)
            temp = end
            print(f'\rDetecting: [{i:02d}] {j}/{len(dataset)}, {fps:.2f} fps.', end='')

        psnr_group.append(np.array(psnrs))
    print('\nAll frames were detected, begin to compute AUC.')

    gt_loader = Label_loader(cfg, video_folders)  # Get gt labels.
    gt = gt_loader()

    assert len(psnr_group) == len(gt), f'Ground truth has {len(gt)} videos, but got {len(psnr_group)} detected videos.'

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    for i in range(len(psnr_group)):
        distance = psnr_group[i]
        distance -= min(distance)  # distance = (distance - min) / (max - min)
        distance /= max(distance)

        scores = np.concatenate((scores, distance), axis=0)
        labels = np.concatenate((labels, gt[i][4:]), axis=0)  # Exclude the first 4 unpredictable frames in gt.

    assert scores.shape == labels.shape, \
        f'Ground truth has {labels.shape[0]} frames, but got {scores.shape[0]} detected frames.'

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    print(f'AUC: {auc}\n')
    return auc


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
