import cv2
import time
import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse

from utils import *
from losses import *
import Dataset
from models.unet import UNet
from models.pix2pix_networks import PixelDiscriminator
from models.liteFlownet import lite_flownet as lite_flow
from config import update_config
from models.flownet2.models import FlowNet2SD
from evaluate import val

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--img_size', default=(256, 256), type=tuple, help='The image size for training and evaluating.')
parser.add_argument('--color_type', default='colorful', type=str, help='The color type of the dataset.')
parser.add_argument('--input_num', default='4', type=int, help='The frame number to be used to predict one frame.')
parser.add_argument('--iters', default=80000, type=int, help='The total iteration number.')
parser.add_argument('--resume', default=None, type=str,
                    help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
parser.add_argument('--save_interval', default=5000, type=int, help='Save the model every [save_interval] iterations.')
parser.add_argument('--val_interval', default=10000, type=int,
                    help='Evaluate the model every [val_interval] iterations, pass -1 to disable.')
parser.add_argument('--show_flow', default=False, action='store_true',
                    help='If True, the first batch of ground truth optic flow could be visualized and saved.')
parser.add_argument('--flownet', default='lite', type=str, help='lite: LiteFlownet, 2sd: FlowNet2SD.')

args = parser.parse_args()
train_cfg = update_config(args, mode='train')
train_cfg.print_cfg()

color_c = 3 if train_cfg.color_type == 'colorful' else 1
generator = UNet(input_channels=color_c * train_cfg.input_num, output_channel=color_c).cuda()
discriminator = PixelDiscriminator(input_nc=color_c).cuda()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=train_cfg.g_lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=train_cfg.d_lr)

if train_cfg.resume:
    generator.load_state_dict(torch.load(train_cfg.resume)['net_g'])
    discriminator.load_state_dict(torch.load(train_cfg.resume)['net_d'])
    optimizer_G.load_state_dict(torch.load(train_cfg.resume)['optimizer_g'])
    optimizer_D.load_state_dict(torch.load(train_cfg.resume)['optimizer_d'])
    print(f'Pre-trained generator and discriminator have been loaded.\n')
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    print('Generator and discriminator are going to be trained from scratch.\n')

assert train_cfg.flownet in ('lite', '2sd'), 'Flow net only supports LiteFlownet or FlowNet2SD currently.'
if train_cfg.flownet == '2sd':
    flow_net = FlowNet2SD()
    flow_net.load_state_dict(torch.load('models/flownet2/FlowNet2-SD.pth')['state_dict'])
else:
    flow_net = lite_flow.Network()
    flow_net.load_state_dict(torch.load('models/liteFlownet/network-default.pytorch'))

flow_net.cuda().eval()  # Use flow_net to generate optic flows, so set to eval mode.

adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gradient_loss = Gradient_Loss(color_c).cuda()
flow_loss = Flow_Loss().cuda()
intensity_loss = Intensity_Loss().cuda()

writer = SummaryWriter('tensorboard_log')

# TODO: the grey image can still be read as 3 channels, how to decide the channel issue?
train_dataset = Dataset.train_dataset(train_cfg)
# Remember to set drop_last=True, because we need to use 4 frames to predict one frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)

step = int(train_cfg.resume.split('_')[-1].split('.')[0]) if train_cfg.resume else 0
elapsed = 0.
training = True
generator = generator.train()
discriminator = discriminator.train()

try:
    while training:
        for clips, flow_strs in train_dataloader:
            step += 1

            input_frames = clips[:, 0:12, :, :].cuda()  # (n, 12, 256, 256)
            target_frame = clips[:, 12:15, :, :].cuda()  # (n, 3, 256, 256)
            input_last = input_frames[:, 9:12, :, :].cuda()  # use for flow_loss

            G_frame = generator(input_frames)

            if train_cfg.flownet == 'lite':
                gt_flow_input = torch.cat([input_last, target_frame], 1)
                pred_flow_input = torch.cat([input_last, G_frame], 1)
                flow_gt = flow_net.batch_estimate(gt_flow_input, flow_net)
                flow_pred = flow_net.batch_estimate(pred_flow_input, flow_net)
                # TODO: only can in lite_flownet now, change it.
                if train_cfg.show_flow:
                    flow = np.array(flow_gt.cpu().detach().numpy().transpose(0, 2, 3, 1), np.float32)  # to (n, w, h, 2)
                    for i in range(flow.shape[0]):
                        aa = flow_to_color(flow[i], convert_to_bgr=False)
                        path = train_cfg.train_data.split('/')[-3] + '_' + flow_strs[i]
                        cv2.imwrite(f'images/{path}.jpg', aa)  # e.g. images/avenue_4_574-575.jpg
                        print(f'Saved a sample optic flow image from gt frames: \'images/{path}.jpg\'.')

            else:
                gt_flow_input = torch.cat([input_last.unsqueeze(2), target_frame.unsqueeze(2)], 2)
                pred_flow_input = torch.cat([input_last.unsqueeze(2), G_frame.unsqueeze(2)], 2)

                flow_gt = flow_net(gt_flow_input * 255.0)  # Input for flownet2sd is in (0, 255).
                flow_pred = flow_net(pred_flow_input * 255.0)

            inte_l = intensity_loss(G_frame, target_frame)
            grad_l = gradient_loss(G_frame, target_frame)
            fl_l = flow_loss(flow_pred, flow_gt)
            g_l = adversarial_loss(discriminator(G_frame))
            G_l_t = 1. * inte_l + 1. * grad_l + 2. * fl_l + 0.05 * g_l

            # Train discriminator
            # When training discriminator, the weights of generator are fixed, so use .detach() to cut off gradients.
            D_l = discriminate_loss(discriminator(target_frame), discriminator(G_frame.detach()))
            optimizer_D.zero_grad()
            D_l.backward()
            optimizer_D.step()

            # Train generator
            optimizer_G.zero_grad()
            G_l_t.backward()
            optimizer_G.step()

            # torch.cuda.synchronize()
            time_end = time.time()
            if step > 1:  # This considers all the consumed time, including the testing time during training.
                iter_t = time_end - temp
                elapsed += iter_t
            temp = time_end

            if step % 20 == 0:
                time_reamin = (train_cfg.iters - step) / step * elapsed
                eta = str(datetime.timedelta(seconds=time_reamin)).split('.')[0]
                psnr = psnr_error(G_frame, target_frame)
                print(f"[{step}]  inte_l: {inte_l:.3f} | grad_l: {grad_l:.3f} | fl_l: {fl_l:.3f} | g_l: {g_l:.3f} | "
                      f"G_l_total: {G_l_t:.3f} | D_l: {D_l:.3f} | psnr: {psnr:.3f} | iter: {iter_t:.3f}s | ETA: {eta}")

                save_G_frame = ((G_frame[0] + 1) * 127.5)
                save_G_frame = save_G_frame.cpu().detach().numpy().astype('uint8')[..., (2, 1, 0)]
                save_target = ((target_frame[0] + 1) * 127.5)
                save_target = save_target.cpu().detach().numpy().astype('uint8')[..., (2, 1, 0)]

                writer.add_scalar('psnr/train_psnr', psnr, global_step=step)
                writer.add_scalar('total_loss/g_loss_total', G_l_t, global_step=step)
                writer.add_scalar('total_loss/d_loss', D_l, global_step=step)
                writer.add_scalar('G_loss_total/g_loss', g_l, global_step=step)
                writer.add_scalar('G_loss_total/fl_loss', fl_l, global_step=step)
                writer.add_scalar('G_loss_total/inte_loss', inte_l, global_step=step)
                writer.add_scalar('G_loss_total/grad_loss', grad_l, global_step=step)
                writer.add_scalar('psnr/train_psnr', psnr, global_step=step)

            if step % int(train_cfg.iters / 100) == 0:
                writer.add_image('image/G_frame', save_G_frame, global_step=step)  # Channel order should be (C, W, H).
                writer.add_image('image/target', save_target, global_step=step)

            if step % train_cfg.save_interval == 0:
                model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                              'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                torch.save(model_dict, f'weights/{train_cfg.dataset}_{step}.pth')
                print(f'\nAlready saved: \'{train_cfg.dataset}_{step}.pth\'.')

            if step % train_cfg.val_interval == 0:
                auc = val(train_cfg, model=generator)
                writer.add_scalar('results/auc', auc, global_step=step)
                generator.train()

            if step > train_cfg.iters:
                training = False
                break

except KeyboardInterrupt:
    print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{step}.pth\'.\n')
    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                  'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
    torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}.pth')
