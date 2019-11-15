import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

# from networks.resample2d_package.resample2d import Resample2d
# from networks.channelnorm_package.channelnorm import ChannelNorm

# from networks import FlowNetC
# from networks import FlowNetS
from flownet2.networks import FlowNetSD
# from networks import FlowNetFusion

from flownet2.networks.submodules import *

'Parameter count = 162,518,834'

#

class FlowNet2SD(FlowNetSD.FlowNetSD):
    def __init__(self,batchNorm=False, div_flow=20):
        super(FlowNet2SD, self).__init__( batchNorm=batchNorm)
        self.rgb_max = 255.
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)

        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)

        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return self.upsample1(flow2 * self.div_flow)


# class FlowNet2CS(nn.Module):
#
#     def __init__(self, args, batchNorm=False, div_flow=20.):
#         super(FlowNet2CS, self).__init__()
#         self.batchNorm = batchNorm
#         self.div_flow = div_flow
#         self.rgb_max = args.rgb_max
#         self.args = args
#
#         self.channelnorm = ChannelNorm()
#
#         # First Block (FlowNetC)
#         self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
#         self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
#
#         if args.fp16:
#             self.resample1 = nn.Sequential(
#                 tofp32(),
#                 Resample2d(),
#                 tofp16())
#         else:
#             self.resample1 = Resample2d()
#
#         # Block (FlowNetS1)
#         self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
#         self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.bias is not None:
#                     init.uniform(m.bias)
#                 init.xavier_uniform(m.weight)
#
#             if isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     init.uniform(m.bias)
#                 init.xavier_uniform(m.weight)
#                 # init_deconv_bilinear(m.weight)
#
#     def forward(self, inputs):
#         rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
#
#         x = (inputs - rgb_mean) / self.rgb_max
#         x1 = x[:, :, 0, :, :]
#         x2 = x[:, :, 1, :, :]
#         x = torch.cat((x1, x2), dim=1)
#
#         # flownetc
#         flownetc_flow2 = self.flownetc(x)[0]
#         flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
#
#         # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
#         resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
#         diff_img0 = x[:, :3, :, :] - resampled_img1
#         norm_diff_img0 = self.channelnorm(diff_img0)
#
#         # concat img0, img1, img1->img0, flow, diff-mag ;
#         concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)
#
#         # flownets1
#         flownets1_flow2 = self.flownets_1(concat1)[0]
#         flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
#
#         return flownets1_flow
#
#
# class FlowNet2CSS(nn.Module):
#
#     def __init__(self, args, batchNorm=False, div_flow=20.):
#         super(FlowNet2CSS, self).__init__()
#         self.batchNorm = batchNorm
#         self.div_flow = div_flow
#         self.rgb_max = args.rgb_max
#         self.args = args
#
#         self.channelnorm = ChannelNorm()
#
#         # First Block (FlowNetC)
#         self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
#         self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
#
#         if args.fp16:
#             self.resample1 = nn.Sequential(
#                 tofp32(),
#                 Resample2d(),
#                 tofp16())
#         else:
#             self.resample1 = Resample2d()
#
#         # Block (FlowNetS1)
#         self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
#         self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
#         if args.fp16:
#             self.resample2 = nn.Sequential(
#                 tofp32(),
#                 Resample2d(),
#                 tofp16())
#         else:
#             self.resample2 = Resample2d()
#
#         # Block (FlowNetS2)
#         self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
#         self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.bias is not None:
#                     init.uniform(m.bias)
#                 init.xavier_uniform(m.weight)
#
#             if isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     init.uniform(m.bias)
#                 init.xavier_uniform(m.weight)
#                 # init_deconv_bilinear(m.weight)
#
#     def forward(self, inputs):
#         rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
#
#         x = (inputs - rgb_mean) / self.rgb_max
#         x1 = x[:, :, 0, :, :]
#         x2 = x[:, :, 1, :, :]
#         x = torch.cat((x1, x2), dim=1)
#
#         # flownetc
#         flownetc_flow2 = self.flownetc(x)[0]
#         flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
#
#         # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
#         resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
#         diff_img0 = x[:, :3, :, :] - resampled_img1
#         norm_diff_img0 = self.channelnorm(diff_img0)
#
#         # concat img0, img1, img1->img0, flow, diff-mag ;
#         concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)
#
#         # flownets1
#         flownets1_flow2 = self.flownets_1(concat1)[0]
#         flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
#
#         # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
#         resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
#         diff_img0 = x[:, :3, :, :] - resampled_img1
#         norm_diff_img0 = self.channelnorm(diff_img0)
#
#         # concat img0, img1, img1->img0, flow, diff-mag
#         concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.div_flow, norm_diff_img0), dim=1)
#
#         # flownets2
#         flownets2_flow2 = self.flownets_2(concat2)[0]
#         flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)
#
#         return flownets2_flow
#
