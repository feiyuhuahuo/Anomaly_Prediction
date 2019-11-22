from utils import *
from losses import *
import img_dataset
from models.unet import UNet
from models.pix2pix_networks import PixelDiscriminator
from liteFlownet import lite_flownet as lite_flow
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
from config import *
from flownet2.models import FlowNet2SD
from evaluate import evaluate

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--resume', default=None, type=str, help='The path of the weight file to resume training with.')
parser.add_argument('--val_interval', default=10000, type=int,
                    help='Evalute and save the model every [val_interval] iterations, pass -1 to disable.')
parser.add_argument('--flownet2sd', default=False, action='store_true', help='Use Flownet2SD instead of LiteFlownet.')

args = parser.parse_args()

generator = UNet(input_channels=12, output_channel=3).cuda()
discriminator = PixelDiscriminator(3, [128, 256, 512, 512], use_norm=False).cuda()

if not args.resume:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
else:
    generator.load_state_dict(torch.load(generator_model))
    discriminator.load_state_dict(torch.load(discriminator_model))
    step = int(generator_model.split('-')[-1])
    print('pretrained model loaded!')

if args.flownet2sd:
    flow_net = FlowNet2SD()
    flow_net.load_state_dict(torch.load(flownet2SD_model_path)['state_dict'])
else:
    flow_net = lite_flow.Network()
    flow_net.load_state_dict(torch.load(liteflow_model))

flow_net.cuda().eval()  # Use FlowNet to generate optic flows, so set to eval mode.

optimizer_G = torch.optim.Adam(generator.parameters(), lr=g_lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gradient_loss = Gradient_Loss(num_channels).cuda()
flow_loss = Flow_Loss().cuda()
intensity_loss = Intensity_Loss().cuda()

writer = SummaryWriter(writer_path)

train_dataset = img_dataset.ano_pred_Dataset(train_data, num_clips)
test_dataset = img_dataset.ano_pred_Dataset(test_data, num_clips)
# Remember to set drop_last=True because we need to use 4 frames to predict one frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

step = 0
for epoch in range(epochs):
    for five_frames in train_dataloader:
        generator = generator.train()
        discriminator = discriminator.train()

        input_frames = five_frames[:, 0:12, :, :].cuda()  # (n, 12, 256, 256)
        target_frame = five_frames[:, 12:15, :, :].cuda()  # (n, 3, 256, 256)
        input_last = input_frames[:, 9:12, :, :].cuda()  # use for flow_loss

        # Re-show the original frames.
        # aa = np.array(gt_frames.permute(0, 2, 3, 1).cpu()) * 255
        # aa = aa.astype('uint8')
        # import cv2
        # for i in range(4):
        #     cv2.imshow('aa', aa[0, :, :, 3*i:3*i+3])
        #     cv2.waitKey()

        G_frame = generator(input_frames)

        if not args.flownet2sd:
            gt_flow_input = torch.cat([input_last, target_frame], 1)
            pred_flow_input = torch.cat([input_last, G_frame], 1)

            flow_gt = flow_net.batch_estimate(gt_flow_input, flow_net)
            flow_pred = flow_net.batch_estimate(pred_flow_input, flow_net)
        else:
            gt_flow_input = torch.cat([input_last.unsqueeze(2), target_frame.unsqueeze(2)], 2)  # (2, 3, 2, 256, 256)
            pred_flow_input = torch.cat([input_last.unsqueeze(2), G_frame.unsqueeze(2)], 2)

            flow_gt = flow_net(gt_flow_input * 255.0)  # Input for flownet2sd is in (0, 255).
            flow_pred = flow_net(pred_flow_input * 255.0)

        inte_loss = intensity_loss(G_frame, target_frame)
        grad_loss = gradient_loss(G_frame, target_frame)
        fl_loss = flow_loss(flow_pred, flow_gt)
        G_loss = adversarial_loss(discriminator(G_frame))
        G_loss_total = 1. * inte_loss + 1. * grad_loss + 2. * fl_loss + 0.05 * G_loss

        # Train discriminator
        # When training discriminator, the weights of generator are fixed, so here use .detach() to cut off gradients.
        D_loss = discriminate_loss(discriminator(target_frame), discriminator(G_frame.detach()))
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad()
        G_loss_total.backward()
        optimizer_G.step()

        if step % 10 == 0:
            train_psnr = psnr_error(G_frame, target_frame)
            print(f"[{step}/{epoch}]: g_loss: {G_loss_total} d_loss {D_loss}")
            print(f'gd_loss {grad_loss}, op_loss {fl_loss}, int_loss {inte_loss}')
            print(f'train psnr{train_psnr}')

            writer.add_scalar('psnr/train_psnr', train_psnr, global_step=step)
            writer.add_scalar('total_loss/g_loss', G_loss_total, global_step=step)
            writer.add_scalar('total_loss/d_loss', D_loss, global_step=step)
            writer.add_scalar('g_loss/adv_loss', G_loss, global_step=step)
            writer.add_scalar('g_loss/op_loss', fl_loss, global_step=step)
            writer.add_scalar('g_loss/int_loss', inte_loss, global_step=step)
            writer.add_scalar('g_loss/gd_loss', grad_loss, global_step=step)

            writer.add_image('image/train_target', target_frame[0], global_step=step)
            writer.add_image('image/train_output', G_frame[0], global_step=step)

        step += 1

        if step % 500 == 0:
            saver(generator.state_dict(), generator_model, step, max_to_save=10)
            saver(discriminator.state_dict(), discriminator_model, step, max_to_save=10)
            if step >= 2000:
                print('==== begin evaluate the model of {} ===='.format(generator_model + '-' + str(step)))

                auc = evaluate(frame_num=5, input_channels=12, output_channels=3,
                               model_path=generator_model + '-' + str(step))
                writer.add_scalar('results/auc', auc, global_step=step)

# if __name__ == '__main__':
#     train(num_clips, num_unet_layers, num_channels * (num_clips - num_his), num_channels, discriminator_channels)

# pretrain=True,
# generator_pretrain_path='../pth_model/ano_pred_avenue_generator_2.pth-4500',
# discriminator_pretrain_path='../pth_model/ano_pred_avenue_discriminator_2.pth-4500')

# test(num_clips,num_unet_layers,num_channels*(num_clips-num_his),num_channels,discriminator_channels)
# _test()
# test(0,0,0,0,0)
