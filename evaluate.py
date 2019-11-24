import Dataset
from models.unet import UNet
from torch.utils.data import DataLoader
import numpy as np
from utils import psnr_error
import os
import time
import pickle
import torch
from config import test_data
from sklearn import metrics
from utils import RecordResult
from Dataset import label_loader

dataset_name = 'avenue'

psnr_dir = '../psnr/'


def compute_auc(loss_file):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        with open(sub_loss_file, 'rb') as reader:
            # results {
            #   'dataset': the name of dataset
            #   'psnr': the psnr of each testing videos,
            # }

            # psnr_records['psnr'] is np.array, shape(#videos)
            # psnr_records[0] is np.array   ------>     01.avi
            # psnr_records[1] is np.array   ------>     02.avi
            #               ......
            # psnr_records[n] is np.array   ------>     xx.avi

            results = pickle.load(reader)

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

        results = RecordResult(fpr, tpr, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results


def evaluate(model_path):
    generator = UNet(input_channels=12, output_channel=3).cuda().eval()
    video_folders = os.listdir(test_data)

    video_folders.sort()

    time_stamp = time.time()

    psnr_records = []
    total = 0
    generator.load_state_dict(torch.load(model_path))

    for folder in video_folders:
        _temp_test_folder = os.path.join(test_data, folder)
        dataset = Dataset.test_dataset(_temp_test_folder, clip_length=frame_num)

        test_iters = len(dataset) - frame_num + 1
        test_counter = 0

        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

        psnrs = np.empty(shape=(len(dataset),), dtype=np.float32)
        for test_input in data_loader:
            input_frames = test_input[:, 0:12, :, :].cuda()
            target_frame = test_input[:, 12:15, :, :].cuda()

            G_frame = generator(input_frames)
            test_psnr = psnr_error(G_frame, target_frame)
            test_psnr = test_psnr.tolist()
            psnrs[test_counter + frame_num - 1] = test_psnr

            test_counter += 1
            total += 1
            if test_counter >= test_iters:
                psnrs[:frame_num - 1] = psnrs[frame_num - 1]
                psnr_records.append(psnrs)
                print('finish test video set {}'.format(_temp_test_folder))
                break

    result_dict = {'dataset': dataset_name, 'psnr': psnr_records, 'flow': [], 'names': [], 'diff_mask': []}

    used_time = time.time() - time_stamp
    print('total time = {}, fps = {}'.format(used_time, total / used_time))

    pickle_path = os.path.join(psnr_dir, os.path.split(model_path)[-1])

    with open(pickle_path, 'wb') as writer:
        pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)

    results = compute_auc(pickle_path)
    print(results)


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
    evaluate(model_path='weights/ped2_90000.pth')
