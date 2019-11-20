import sys
sys.path.append('..')
from Dataset import img_dataset
from models.unet import UNet
from torch.utils.data import DataLoader
import numpy as np
from ano_pre.util import psnr_error
import pdb
import os
import time
import pickle
from ano_pre import eval_metric


testing_data_folder = '/home/feiyu/Data/avenue/testing/frames'
dataset_name = 'avenue'

psnr_dir = '../psnr/'


def evaluate(frame_num, input_channels, output_channels, model_path, evaluate_name):
    generator = UNet(input_channels=input_channels, output_channel=output_channels).cuda().eval()
    video_folders = os.listdir(testing_data_folder)

    video_folders.sort()

    time_stamp = time.time()

    psnr_records = []
    total = 0
    # generator.load_state_dict(torch.load(model_path))

    for folder in video_folders:
        _temp_test_folder = os.path.join(testing_data_folder, folder)
        dataset = img_dataset.test_dataset(_temp_test_folder, clip_length=frame_num)

        test_iters = len(dataset) - frame_num + 1
        test_counter = 0

        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

        psnrs = np.empty(shape=(len(dataset),), dtype=np.float32)
        for test_input in data_loader:
            input_frames = test_input[:, 0:12, :, :].cuda()
            target_frame = test_input[:, 12:15, :, :].cuda()

            G_frame = generator(input_frames)
            test_psnr = psnr_error(G_frame, target_frame)
            print(test_psnr)
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

    results = eval_metric.compute_auc(pickle_path)
    print(results)


if __name__ == '__main__':
    evaluate(frame_num=5, input_channels=12, output_channels=3,
             model_path='../pth_model/ano_pred_avenue_generator.pth-9000', evaluate_name='compute_auc')
