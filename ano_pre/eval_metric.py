import numpy as np
import scipy.io as scio
import os
import pickle
from sklearn import metrics
import json
import pdb


DATA_DIR = '/home/feiyu/Data/'

# normalize scores in each sub video
NORMALIZE = True

# number of history frames, since in prediction based method, the first 4 frames can not be predicted, so that
# the first 4frames are undecidable, we just ignore the first 4 frames
DECIDABLE_IDX = 4


class RecordResult(object):
    def __init__(self, fpr=None, tpr=None, auc=-np.inf, dataset=None, loss_file=None):
        self.fpr = fpr
        self.tpr = tpr
        self.auc = auc
        self.dataset = dataset
        self.loss_file = loss_file

    def __lt__(self, other):
        return self.auc < other.auc

    def __gt__(self, other):
        return self.auc > other.auc

    def __str__(self):
        return 'dataset = {}, loss file = {}, auc = {}'.format(self.dataset, self.loss_file, self.auc)


class GroundTruthLoader(object):
    AVENUE = 'avenue'
    PED1 = 'ped1'
    PED2 = 'ped2'

    SHANGHAITECH = 'shanghaitech'
    SHANGHAITECH_LABEL_PATH = os.path.join(DATA_DIR, 'shanghaitech/testing/test_frame_mask')

    NAME_MAT_MAPPING = {AVENUE: os.path.join(DATA_DIR, 'avenue/avenue.mat'),
                        PED1: os.path.join(DATA_DIR, 'ped1/ped1.mat'),
                        PED2: os.path.join(DATA_DIR, 'ped2/ped2.mat')}

    NAME_FRAMES_MAPPING = {AVENUE: os.path.join(DATA_DIR, 'avenue/testing/frames'),
                           PED1: os.path.join(DATA_DIR, 'ped1/testing/frames'),
                           PED2: os.path.join(DATA_DIR, 'ped2/testing/frames')}

    def __init__(self):
        self.mapping = GroundTruthLoader.NAME_MAT_MAPPING

    def __call__(self, dataset):
        """ get the ground truth by provided the name of dataset.

        :type dataset: str
        :param dataset: the name of dataset.
        :return: np.ndarray, shape(#video)
                 np.array[0] contains all the start frame and end frame of abnormal events of video 0,
                 and its shape is (#frapsnr, )
        """

        if dataset == GroundTruthLoader.SHANGHAITECH:
            gt = self.__load_shanghaitech_gt()
        else:
            gt = self.load_ucsd_avenue_gt(dataset)
        return gt

    def load_ucsd_avenue_gt(self, dataset):
        assert dataset in self.mapping, 'there is no dataset named {} \n Please check {}' \
            .format(dataset, GroundTruthLoader.NAME_MAT_MAPPING.keys())

        mat_file = self.mapping[dataset]
        abnormal_events = scio.loadmat(mat_file, squeeze_me=True)['gt']

        if abnormal_events.ndim == 2:
            abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0], abnormal_events.shape[1])

        num_video = abnormal_events.shape[0]
        dataset_video_folder = GroundTruthLoader.NAME_FRAMES_MAPPING[dataset]
        video_list = os.listdir(dataset_video_folder)
        video_list.sort()

        assert num_video == len(video_list), 'ground true does not match the number of testing videos. {} != {}' \
            .format(num_video, len(video_list))

        # get the total frames of sub video
        def get_video_length(sub_video_number):
            # video_name = video_name_template.format(sub_video_number)
            video_name = os.path.join(dataset_video_folder, video_list[sub_video_number])
            assert os.path.isdir(video_name), '{} is not directory!'.format(video_name)

            length = len(os.listdir(video_name))

            return length

        # need to test [].append, or np.array().append(), which one is faster
        gt = []
        for i in range(num_video):
            length = get_video_length(i)

            sub_video_gt = np.zeros((length,), dtype=np.int8)
            sub_abnormal_events = abnormal_events[i]
            if sub_abnormal_events.ndim == 1:
                sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))

            _, num_abnormal = sub_abnormal_events.shape

            for j in range(num_abnormal):
                # (start - 1, end - 1)
                start = sub_abnormal_events[0, j] - 1
                end = sub_abnormal_events[1, j]

                sub_video_gt[start: end] = 1

            gt.append(sub_video_gt)

        return gt

    @staticmethod
    def __load_shanghaitech_gt():
        video_path_list = os.listdir(GroundTruthLoader.SHANGHAITECH_LABEL_PATH)
        video_path_list.sort()

        gt = []
        for video in video_path_list:
            # print(os.path.join(GroundTruthLoader.SHANGHAITECH_LABEL_PATH, video))
            gt.append(np.load(os.path.join(GroundTruthLoader.SHANGHAITECH_LABEL_PATH, video)))

        return gt


def load_psnr_gt(loss_file):
    with open(loss_file, 'rb') as reader:
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
    gt_loader = GroundTruthLoader()
    gt = gt_loader(dataset=dataset)

    assert num_videos == len(gt), 'the number of saved videos does not match the ground truth, {} != {}' \
        .format(num_videos, len(gt))

    return dataset, psnr_records, gt


def compute_auc(loss_file):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        # the name of dataset, loss, and ground truth
        dataset, psnr_records, gt = load_psnr_gt(loss_file=sub_loss_file)

        # the number of videos
        num_videos = len(psnr_records)

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        # video normalization
        for i in range(num_videos):
            distance = psnr_records[i]

            #             if NORMALIZE:
            #                 distance -= distance.min()  # distances = (distance - min) / (max - min)
            #                 distance /= distance.max()
            # distance = 1 - distance

            scores = np.concatenate((scores, distance[DECIDABLE_IDX:]), axis=0)
            labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)
        if NORMALIZE:
            scores -= scores.min()  # scores = (scores - min) / (max - min)
            scores /= scores.max()
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)

        results = RecordResult(fpr, tpr, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results


# labels = [0,  0,   0,   0,   0,  1,   1,    1,   0,  1,   0,    0]
# scores = [0, 1/8, 2/8, 1/8, 1/8, 3/8, 6/8, 7/8, 5/8, 8/8, 2/8, 1/8]
# fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
# print(fpr)
# print('~~~~~~~~~~~~`')
# print(tpr)
# print('~~~~~~~~~~~~`')
# print(thresholds)
# print('~~~~~~~~~~~~`')