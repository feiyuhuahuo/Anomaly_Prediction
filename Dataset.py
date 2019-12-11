import random
import torch
import numpy as np
import cv2
import glob
import os
import scipy.io as scio
from torch.utils.data import Dataset


def np_load_frame(filename, resize_h, resize_w):
    img = cv2.imread(filename)
    image_resized = cv2.resize(img, (resize_w, resize_h)).astype('float32')
    image_resized = (image_resized / 127.5) - 1.0  # to -1 ~ 1
    image_resized = np.transpose(image_resized, [2, 0, 1])  # to (C, W, H)
    return image_resized


class train_dataset(Dataset):
    """
    No data augmentation.
    Normalized from [0, 255] to [-1, 1], the channels are BGR due to cv2 and liteFlownet.
    """

    def __init__(self, cfg):
        self.img_h = cfg.img_size[0]
        self.img_w = cfg.img_size[1]
        self.clip_length = 5

        self.videos = []
        self.all_seqs = []
        for folder in sorted(glob.glob(f'{cfg.train_data}/*')):
            all_imgs = glob.glob(f'{folder}/*.jpg')
            all_imgs.sort()
            self.videos.append(all_imgs)

            random_seq = list(range(len(all_imgs) - 4))
            random.shuffle(random_seq)
            self.all_seqs.append(random_seq)

    def __len__(self):  # This decide the indice range of the PyTorch Dataloader.
        return len(self.videos)

    def __getitem__(self, indice):  # Indice decide which video folder to be loaded.
        one_folder = self.videos[indice]

        video_clip = []
        start = self.all_seqs[indice][-1]  # Always use the last index in self.all_seqs.
        for i in range(start, start + self.clip_length):
            video_clip.append(np_load_frame(one_folder[i], self.img_h, self.img_w))

        video_clip = np.array(video_clip).reshape((-1, self.img_h, self.img_w))
        video_clip = torch.from_numpy(video_clip)

        flow_str = f'{indice}_{start + 3}-{start + 4}'
        return indice, video_clip, flow_str


class test_dataset:
    def __init__(self, cfg, video_folder):
        self.img_h = cfg.img_size[0]
        self.img_w = cfg.img_size[1]
        self.clip_length = 5
        self.imgs = glob.glob(video_folder + '/*.jpg')
        self.imgs.sort()

    def __len__(self):
        return len(self.imgs) - (self.clip_length - 1)  # The first [input_num] frames are unpredictable.

    def __getitem__(self, indice):
        video_clips = []
        for frame_id in range(indice, indice + self.clip_length):
            video_clips.append(np_load_frame(self.imgs[frame_id], self.img_h, self.img_w))

        video_clips = np.array(video_clips).reshape((-1, self.img_h, self.img_w))
        return video_clips


class Label_loader:
    def __init__(self, cfg, video_folders):
        assert cfg.dataset in ('ped2', 'avenue', 'shanghaitech'), f'Did not find the related gt for \'{cfg.dataset}\'.'
        self.cfg = cfg
        self.name = cfg.dataset
        self.frame_path = cfg.test_data
        self.mat_path = f'{cfg.data_root + self.name}/{self.name}.mat'
        self.video_folders = video_folders

    def __call__(self):
        if self.name == 'shanghaitech':
            gt = self.load_shanghaitech()
        else:
            gt = self.load_ucsd_avenue()
        return gt

    def load_ucsd_avenue(self):
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

    def load_shanghaitech(self):
        np_list = glob.glob(f'{self.cfg.data_root + self.name}/frame_masks/')
        np_list.sort()

        gt = []
        for npy in np_list:
            gt.append(np.load(npy))

        return gt
