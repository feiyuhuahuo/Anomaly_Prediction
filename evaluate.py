import Dataset
from models.unet import UNet
from torch.utils.data import DataLoader
import numpy as np
from utils import psnr_error
import os
import time
import torch
import argparse
from config import update_config
from sklearn import metrics
from Dataset import label_loader

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--dataset_type', default='colorful', type=str, help='The color type of the dataset.')
parser.add_argument('--model', default='colorful', type=str, help='The pre-trained model to evaluate.')

args = parser.parse_args()
cfg = update_config(args)


def evaluate(cfg):
    generator = UNet(input_channels=12, output_channel=3).cuda().eval()
    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()

    time_stamp = time.time()

    psnr_records = []
    total = 0
    generator.load_state_dict(torch.load(args.model))

    for folder in video_folders:
        _temp_test_folder = os.path.join(cfg.test_data, folder)
        dataset = Dataset.test_dataset(_temp_test_folder, clip_length=5)

        test_iters = len(dataset) - 5 + 1
        test_counter = 0

        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

        psnrs = np.empty(shape=(len(dataset),), dtype=np.float32)
        for test_input in data_loader:
            input_frames = test_input[:, 0:12, :, :].cuda()
            target_frame = test_input[:, 12:15, :, :].cuda()

            G_frame = generator(input_frames)
            test_psnr = psnr_error(G_frame, target_frame)
            test_psnr = test_psnr.tolist()
            psnrs[test_counter + 5 - 1] = test_psnr

            test_counter += 1
            total += 1
            if test_counter >= test_iters:
                psnrs[:5 - 1] = psnrs[5 - 1]
                psnr_records.append(psnrs)
                print('finish test video set {}'.format(_temp_test_folder))
                break

    results = {'dataset': dataset_name, 'psnr': psnr_records, 'diff_mask': []}

    used_time = time.time() - time_stamp
    print('total time = {}, fps = {}'.format(used_time, total / used_time))

    dataset = results['dataset']
    psnr_records = results['psnr']

    num_videos = len(psnr_records)

    # load ground truth
    gt_loader = label_loader()
    gt = gt_loader(dataset=dataset)

    assert num_videos == len(gt), 'the number of saved videos does not match the ground truth, {} != {}' \
        .format(num_videos, len(gt))

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)

    for i in range(num_videos):
        distance = psnr_records[i]
        distance -= distance.min()  # distances = (distance - min) / (max - min)
        distance /= distance.max()

        scores = np.concatenate((scores, distance[4:]), axis=0)  # The first 4 frames are unpredictable.
        labels = np.concatenate((labels, gt[i][4:]), axis=0)

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    print(auc)


if __name__ == '__main__':
    # labels = [0,  0,   0,   0,   0,  1,   1,    1,   0,  1,   0,    0]
    # scores = [0, 1/8, 2/8, 1/8, 1/8, 3/8, 6/8, 7/8, 5/8, 8/8, 2/8, 1/8]
    # fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    # print(fpr)
    # print('~~~~~~~~~~~~`')
    # print(tpr)
    # print('~~~~~~~~~~~~`')
    # print(thresholds)
    # print('~~~~~~~~~~~~`')
    evaluate(cfg)
