import torch
import numpy as np
import cv2
from collections import OrderedDict
import glob
import os
import scipy.io as scio
from torch.utils.data import Dataset

rng = np.random.RandomState(2017)


def np_load_frame(filename, resize_height, resize_width):
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = image_resized / 255.0
    image_resized = np.transpose(image_resized, [2, 0, 1])
    return image_resized


class train_dataset(Dataset):
    """
    No data augmentation.
    Normalized from [0,255] to [0,1], the channels are BGR due to cv2 and liteFlownet.
    """

    # video clip mean
    def __init__(self, dataset_folder, clip_length, size=(256, 256)):
        self.dir = dataset_folder
        self.videos = OrderedDict()
        self.image_height = size[0]
        self.image_width = size[1]
        self.clip_length = clip_length
        self.setup()

    def __len__(self):
        return len(self.videos)

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
        self.videos_keys = self.videos.keys()

    def __getitem__(self, indice):
        # When getting frames, 5 frames are one unit, shuffle across all video folders and all frames in one folder.
        key = list(self.videos_keys)[indice]
        start = rng.randint(0, self.videos[key]['length'] - self.clip_length)
        video_clip = []

        for frame_id in range(start, start + self.clip_length):
            video_clip.append(np_load_frame(self.videos[key]['frame'][frame_id], self.image_height, self.image_width))

        video_clip = np.array(video_clip).reshape((-1, self.image_height, self.image_width))
        video_clip = torch.from_numpy(video_clip)

        return video_clip


class test_dataset(Dataset):
    def __init__(self, video_folder, clip_length, size=(256, 256)):
        self.path = video_folder
        self.clip_length = clip_length
        self.img_height, self.img_width = size
        self.pics = glob.glob(self.path + '/*.jpg')
        self.pics.sort()

    def __len__(self):
        return len(self.pics) - 4

    def __getitem__(self, indice):
        video_clips = []
        for frame_id in range(indice, indice + self.clip_length):
            video_clips.append(np_load_frame(self.pics[frame_id], self.img_height, self.img_width))

        video_clips = np.array(video_clips).reshape((-1, self.img_height, self.img_width))
        video_clips = torch.from_numpy(video_clips)
        return video_clips


class label_loader(object):
    DATA_DIR = '/home/feiyu/Data/'
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
        self.mapping = label_loader.NAME_MAT_MAPPING

    def __call__(self, dataset):
        """ get the ground truth by provided the name of dataset.

        :type dataset: str
        :param dataset: the name of dataset.
        :return: np.ndarray, shape(#video)
                 np.array[0] contains all the start frame and end frame of abnormal events of video 0,
                 and its shape is (#frapsnr, )
        """

        if dataset == label_loader.SHANGHAITECH:
            gt = self.__load_shanghaitech_gt()
        else:
            gt = self.load_ucsd_avenue_gt(dataset)
        return gt

    def load_ucsd_avenue_gt(self, dataset):
        assert dataset in self.mapping, 'there is no dataset named {} \n Please check {}' \
            .format(dataset, label_loader.NAME_MAT_MAPPING.keys())

        mat_file = self.mapping[dataset]
        abnormal_events = scio.loadmat(mat_file, squeeze_me=True)['gt']

        if abnormal_events.ndim == 2:
            abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0], abnormal_events.shape[1])

        num_video = abnormal_events.shape[0]
        dataset_video_folder = label_loader.NAME_FRAMES_MAPPING[dataset]
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
        video_path_list = os.listdir(label_loader.SHANGHAITECH_LABEL_PATH)
        video_path_list.sort()

        gt = []
        for video in video_path_list:
            # print(os.path.join(GroundTruthLoader.SHANGHAITECH_LABEL_PATH, video))
            gt.append(np.load(os.path.join(label_loader.SHANGHAITECH_LABEL_PATH, video)))

        return gt
