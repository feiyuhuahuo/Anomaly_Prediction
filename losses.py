import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np


class Flow_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_flows, gt_flows):
        return torch.mean(torch.abs(gen_flows - gt_flows))


class Intensity_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_frames, gt_frames):
        return torch.mean(torch.abs((gen_frames - gt_frames) ** 2))


class Gradient_Loss(nn.Module):
    def __init__(self, channels):
        super().__init__()

        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos
        # Note: when doing conv2d, the channel order is different from tensorflow, so do permutation.
        self.filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1).cuda()
        self.filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1).cuda()

    def forward(self, gen_frames, gt_frames):
        # Do padding to match the  result of the original tensorflow implementation
        gen_frames_x = nn.functional.pad(gen_frames, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(gen_frames, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(gt_frames, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(gt_frames, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)


class Adversarial_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake_outputs):
        # TODO: compare with torch.nn.MSELoss ?
        return torch.mean((fake_outputs - 1) ** 2 / 2)


class Discriminate_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_outputs, fake_outputs):
        return torch.mean((real_outputs - 1) ** 2 / 2) + torch.mean(fake_outputs ** 2 / 2)


# if __name__ == '__main__':
#     # Debug Gradient_Loss, mainly on the padding issue.
#     import numpy as np
#
#     aa = torch.tensor([[1, 2, 3, 4, 2],
#                        [11, 12, 13, 14, 12],
#                        [1, 2, 3, 4, 2],
#                        [21, 22, 23, 24, 22],
#                        [1, 2, 3, 4, 2]], dtype=torch.float32)
#
#     aa = aa.repeat(4, 3, 1, 1)
#
#     pos = torch.from_numpy(np.identity(3, dtype=np.float32))
#     neg = -1 * pos
#     filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1)
#     filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1)
#
#     gen_frames_x = nn.functional.pad(aa, [0, 1, 0, 0])
#     gen_frames_y = nn.functional.pad(aa, [0, 0, 0, 1])
#
#     gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, filter_x))
#     gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, filter_y))
#
#
#     print(aa)
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#     print(filter_y)  # (2, 1, 3, 3)
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#     print(gen_dx)
