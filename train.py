from utils import *
from losses import *
import Dataset
from models.unet import UNet
from models.pix2pix_networks import PixelDiscriminator
from liteFlownet import lite_flownet as lite_flow
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
import time
import datetime
from config import update_config
from flownet2.models import FlowNet2SD
import cv2

# from evaluate import evaluate

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--dataset_type', default='colorful', type=str, help='The color type of the dataset.')
parser.add_argument('--resume_g', default=None, type=str,
                    help='The path of the pre-trained generator model to resume training with.')
parser.add_argument('--resume_d', default=None, type=str,
                    help='The path of the pre-trained discriminator model to resume training with.')
parser.add_argument('--input_num', default='4', type=int, help='The frame number to be used to predict one frame.')
parser.add_argument('--iters', default='80000', type=int, help='The total iteration number.')
parser.add_argument('--show_flow', default=False, action='store_true',
                    help='If True, the first batch of ground truth optic flow could be visualized and saved.')
parser.add_argument('--val_interval', default=10000, type=int,
                    help='Evalute and save the model every [val_interval] iterations, pass -1 to disable.')
parser.add_argument('--flownet2sd', default=False, action='store_true', help='Use Flownet2SD instead of LiteFlownet.')

args = parser.parse_args()
cfg = update_config(args)
cfg.print_cfg()

color_c = 3 if cfg.dataset_type == 'color' else 1
generator = UNet(input_channels=color_c * cfg.input_num, output_channel=color_c).cuda()
discriminator = PixelDiscriminator(input_nc=color_c).cuda()

if args.resume_g:
    generator.load_state_dict(torch.load(args.resume_g))
    print(f'Pre-trained generator loaded with {args.resume_g}.')
else:
    generator.apply(weights_init_normal)
    print('Generator is going to be trained from scratch.')
if args.resume_d:
    discriminator.load_state_dict(torch.load(args.resume_d))
    print(f'Pre-trained discriminator loaded with {args.resume_d}.')
else:
    discriminator.apply(weights_init_normal)
    print('Discriminator is going to be trained from scratch.')

if args.flownet2sd:
    flow_net = FlowNet2SD()
    flow_net.load_state_dict(torch.load('flownet2/FlowNet2-SD.pth')['state_dict'])
else:
    flow_net = lite_flow.Network()
    flow_net.load_state_dict(torch.load('liteFlownet/network-default.pytorch'))

flow_net.cuda().eval()  # Use flow_net to generate optic flows, so set to eval mode.

optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.g_lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.d_lr)
adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gradient_loss = Gradient_Loss(color_c).cuda()
flow_loss = Flow_Loss().cuda()
intensity_loss = Intensity_Loss().cuda()

writer = SummaryWriter('tensorboard_log')

# TODO: the grey image can still be read as 3 channels, how to decide the channel issue?
train_dataset = Dataset.train_dataset(cfg.train_data, cfg.input_num + 1)
test_dataset = Dataset.train_dataset(cfg.test_data, cfg.input_num + 1)
# Remember to set drop_last=True, because we need to use 4 frames to predict one frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, drop_last=True)

step = 0
training = True
elapsed = 0.
time_start = 0.
while training:
    for five_frames, flow_strs in train_dataloader:
        step += 1

        generator = generator.train()
        discriminator = discriminator.train()

        input_frames = five_frames[:, 0:12, :, :].cuda()  # (n, 12, 256, 256)
        target_frame = five_frames[:, 12:15, :, :].cuda()  # (n, 3, 256, 256)
        input_last = input_frames[:, 9:12, :, :].cuda()  # use for flow_loss

        G_frame = generator(input_frames)

        if not args.flownet2sd:
            gt_flow_input = torch.cat([input_last, target_frame], 1)
            pred_flow_input = torch.cat([input_last, G_frame], 1)
            flow_gt = flow_net.batch_estimate(gt_flow_input, flow_net)
            flow_pred = flow_net.batch_estimate(pred_flow_input, flow_net)
            # TODO: only can in lite_flownet now, change it.
            if args.show_flow:
                flow = np.array(flow_gt.cpu().detach().numpy().transpose(0, 2, 3, 1), np.float32)  # to (n, w, h, 2)
                for i in range(flow.shape[0]):
                    aa = flow_to_color(flow[i], convert_to_bgr=False)
                    path = cfg.train_data.split('/')[-3] + '_' + flow_strs[i]
                    cv2.imwrite(f'images/{path}.jpg', aa)  # e.g. images/avenue_4_574-575.jpg

        else:
            gt_flow_input = torch.cat([input_last.unsqueeze(2), target_frame.unsqueeze(2)], 2)  # (2, 3, 2, 256, 256)
            pred_flow_input = torch.cat([input_last.unsqueeze(2), G_frame.unsqueeze(2)], 2)

            flow_gt = flow_net(gt_flow_input * 255.0)  # Input for flownet2sd is in (0, 255).
            flow_pred = flow_net(pred_flow_input * 255.0)

        inte_l = intensity_loss(G_frame, target_frame)
        grad_l = gradient_loss(G_frame, target_frame)
        fl_l = flow_loss(flow_pred, flow_gt)
        g_l = adversarial_loss(discriminator(G_frame))
        G_l_total = 1. * inte_l + 1. * grad_l + 2. * fl_l + 0.05 * g_l

        # Train discriminator
        # When training discriminator, the weights of generator are fixed, so here use .detach() to cut off gradients.
        D_l = discriminate_loss(discriminator(target_frame), discriminator(G_frame.detach()))
        optimizer_D.zero_grad()
        D_l.backward()
        optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad()
        G_l_total.backward()
        optimizer_G.step()

        time_end = time.time()
        if step > 1:  # This considers all the consumed time, including the testing time during training.
            iter_t = time_end - temp
            elapsed += iter_t
        temp = time_end

        if step % 10 == 0:
            time_reamin = (cfg.iters - step) / step * elapsed
            eta = str(datetime.timedelta(seconds=time_reamin)).split('.')[0]
            psnr = psnr_error(G_frame, target_frame)
            print(f"[{step}]  inte_l: {inte_l:.3f} | grad_l: {grad_l:.3f} | fl_l: {fl_l:.3f} | g_l: {g_l:.3f} | "
                  f"G_l_total: {G_l_total:.3f} | D_l: {D_l:.3f} | psnr: {psnr:.3f} | iter: {iter_t:.3f} | ETA: {eta}")

            writer.add_scalar('psnr/train_psnr', psnr, global_step=step)
            writer.add_scalar('total_loss/g_loss_total', G_l_total, global_step=step)
            writer.add_scalar('total_loss/d_loss', D_l, global_step=step)
            writer.add_scalar('G_loss_total/g_loss', g_l, global_step=step)
            writer.add_scalar('G_loss_total/fl_loss', fl_l, global_step=step)
            writer.add_scalar('G_loss_total/inte_loss', inte_l, global_step=step)
            writer.add_scalar('G_loss_total/grad_loss', grad_l, global_step=step)
            writer.add_image('image/train_target', target_frame[0], global_step=step)
            writer.add_image('image/train_output', G_frame[0], global_step=step)

        if step % 50 == 0:
            torch.save({'net': generator.state_dict(), 'opti': optimizer_G.state_dict()}, f'weights/G_{step}.pth')
            torch.save({'net': discriminator.state_dict(), 'opti': optimizer_D.state_dict()}, f'weights/D_{step}.pth')

        if step > args.iters:
            training = False
            break
            #     if step >= 2000:
            #         print('==== begin evaluate the model of {} ===='.format(generator_model + '-' + str(step)))
            #
            #         auc = evaluate(frame_num=5, input_channels=12, output_channels=3,
            #                        model_path=generator_model + '-' + str(step))
            #         writer.add_scalar('results/auc', auc, global_step=step)
