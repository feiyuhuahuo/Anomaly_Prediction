import torch
import numpy as np
import cv2
import glob
import os
import scipy.io as scio
from torch.utils.data import Dataset


def np_load_frame(filename, resize_height, resize_width):
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height)).astype('float32')
    image_resized = (image_resized / 127.5) - 1.0  # to -1 ~ 1
    image_resized = np.transpose(image_resized, [2, 0, 1])  # to (C, W, H)
    return image_resized


class train_dataset(Dataset):
    """
    No data augmentation.
    Normalized from [0, 255] to [-1, 1], the channels are BGR due to cv2 and liteFlownet.
    """

    def __init__(self, dataset_folder, clip_length, size=(256, 256)):
        self.image_height = size[0]
        self.image_width = size[1]
        self.clip_length = clip_length

        self.videos = []
        for folder in sorted(glob.glob(f'{dataset_folder}/*')):
            all_imgs = glob.glob(f'{folder}/*.jpg')
            all_imgs.sort()
            self.videos.append(all_imgs)

    def __len__(self):  # This decide the indice range of the PyTorch Dataloader.
        return len(self.videos)

    def __getitem__(self, indice):
        # When getting frames, 5 frames are one unit, shuffle across all video folders and all frames in one folder.
        one_folder = self.videos[indice]
        start = np.random.randint(0, len(one_folder) - self.clip_length)
        video_clip = []

        for i in range(start, start + self.clip_length):
            video_clip.append(np_load_frame(one_folder[i], self.image_height, self.image_width))

        video_clip = np.array(video_clip).reshape((-1, self.image_height, self.image_width))
        video_clip = torch.from_numpy(video_clip)

        flow_str = f'{indice}_{start + 3}-{start + 4}'
        return video_clip, flow_str


class test_dataset:
    def __init__(self, video_folder, clip_length, size=(256, 256)):
        self.clip_length = clip_length
        self.img_height, self.img_width = size
        self.imgs = glob.glob(video_folder + '/*.jpg')
        self.imgs.sort()

    def __len__(self):
        return len(self.imgs) - 4  # The first 4 frames are unpredictable, so here minus 4.

    def __getitem__(self, indice):
        video_clips = []
        for frame_id in range(indice, indice + self.clip_length):
            video_clips.append(np_load_frame(self.imgs[frame_id], self.img_height, self.img_width))

        video_clips = np.array(video_clips).reshape((-1, self.img_height, self.img_width))
        video_clips = torch.from_numpy(video_clips)
        return video_clips


class Label_loader:

    # SHANGHAITECH_LABEL_PATH = os.path.join(DATA_DIR, 'shanghaitech/testing/test_frame_mask')
    #
    # NAME_MAT_MAPPING = {AVENUE: os.path.join(DATA_DIR, 'avenue/avenue.mat'),
    #                     PED1: os.path.join(DATA_DIR, 'ped1/ped1.mat'),
    #                     PED2: os.path.join(DATA_DIR, 'ped2/ped2.mat')}
    #
    # NAME_FRAMES_MAPPING = {AVENUE: os.path.join(DATA_DIR, 'avenue/testing/frames'),
    #                        PED1: os.path.join(DATA_DIR, 'ped1/testing/frames'),
    #                        PED2: os.path.join(DATA_DIR, 'ped2/testing/frames')}

    def __init__(self, cfg, video_folders):
        assert cfg.dataset in ('ped2', 'avenue', 'shanghaitech'), f'Did not find the related gt for \'{cfg.dataset}\'.'
        self.name = cfg.dataset
        self.frame_path = cfg.test_data
        self.mat_path = f'{cfg.data_root + self.name}/{self.name}.mat'
        self.video_folders = video_folders

    def __call__(self):
        if self.name == 'shanghaitech':
            gt = self.__load_shanghaitech_gt()
        else:
            gt = self.load_ucsd_avenue_gt()
        return gt

    def load_ucsd_avenue_gt(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']

        all_gt = []
        for i in range(abnormal_events.shape[0]):
            length = len(os.listdir(self.video_folders[i]))
            sub_video_gt = np.zeros((length,), dtype=np.int8)

            one_abnormal = abnormal_events[i]
            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]

                sub_video_gt[start: end] = 1

            all_gt.append(sub_video_gt)

        return all_gt

    @staticmethod
    def __load_shanghaitech_gt():
        video_path_list = os.listdir(Label_loader.SHANGHAITECH_LABEL_PATH)
        video_path_list.sort()

        gt = []
        for video in video_path_list:
            # print(os.path.join(GroundTruthLoader.SHANGHAITECH_LABEL_PATH, video))
            gt.append(np.load(os.path.join(Label_loader.SHANGHAITECH_LABEL_PATH, video)))

        return gt
