import sys

sys.path.append('..')
from ano_pre.util import psnr_error
from ano_pre.losses import *
from Dataset import img_dataset
from models.unet import UNet
from models.pix2pix_networks import PixelDiscriminator
from liteFlownet import lite_flownet as lite_flow
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import utils
from ano_pre.evaluate import evaluate

# your gpu id
torch.cuda.set_device(0)

train_data = '/home/feiyu/Data/avenue/training/frames'
test_data = '/home/feiyu/Data/avenue/testing/frames'

generator_model = '../pth_model/ano_pred_avenue_generator_2.pth'
discriminator_model = '../pth_model/ano_pred_avenue_discriminator_2.pth'
liteflow_model = '../liteFlownet/network-default.pytorch'

writer_path = '../log/ano_pred_avenue'

batch_size = 4
epochs = 20000
pretrain = False

# color dataset
g_lr = 0.0002
d_lr = 0.00002

# different range with the source version, should change
lam_int = 1.0 * 2
lam_gd = 1.0 * 2
# here we use no flow loss
lam_op = 0  # 2.0

lam_adv = 0.05

# for gradient loss
alpha = 1
# for int loss
l_num = 2

num_clips = 5
num_his = 1
num_unet_layers = 4

num_channels = 3  # avenue is 3, UCSD is 1
discriminator_channels = [128, 256, 512, 512]


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


generator = UNet(input_channels=12, output_channel=3)
discriminator = PixelDiscriminator(3, [128, 256, 512, 512], use_norm=False)

generator = generator.cuda()
discriminator = discriminator.cuda()

flow_network = lite_flow.Network()
flow_network.load_state_dict(torch.load(liteflow_model))
flow_network.cuda().eval()  # Use liteFlownet to generate optic flows, so set to eval mode.

# if you want to use flownet2SD, comment out the part in front
# flow_network=FlowNet2SD().cuda().eval()
# flow_network.load_state_dict(torch.load(flownet2SD_model_path)['state_dict'])

adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gd_loss = Gradient_Loss(alpha, num_channels).cuda()
op_loss = Flow_Loss().cuda()
int_loss = Intensity_Loss(l_num).cuda()
step = 0

if not pretrain:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
else:
    generator.load_state_dict(torch.load(generator_model))
    discriminator.load_state_dict(torch.load(discriminator_model))
    step = int(generator_model.split('-')[-1])
    print('pretrained model loaded!')

optimizer_G = torch.optim.Adam(generator.parameters(), lr=g_lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=d_lr)

writer = SummaryWriter(writer_path)

dataset = img_dataset.ano_pred_Dataset(train_data, num_clips)
train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

test_dataset = img_dataset.ano_pred_Dataset(test_data, num_clips)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                             drop_last=True)

for epoch in range(epochs):
    for (input, test_input) in zip(train_dataloader, test_dataloader):
        generator = generator.train()
        discriminator = discriminator.train()

        target = input[:, -1, :, :, :].cuda()

        input = input[:, :-1, ]
        input_last = input[:, -1, ].cuda()
        input = input.view(input.shape[0], -1, input.shape[-2], input.shape[-1]).cuda()

        test_target = test_input[:, -1, ].cuda()
        test_input = test_input[:, :-1].view(test_input.shape[0], -1, test_input.shape[-2],
                                             test_input.shape[-1]).cuda()

        # ------- update optim_G --------------
        G_output = generator(input)

        pred_flow_esti_tensor = torch.cat([input_last, G_output], 1)
        gt_flow_esti_tensor = torch.cat([input_last, target], 1)
        flow_gt = lite_flow.batch_estimate(gt_flow_esti_tensor, flow_network)
        flow_pred = lite_flow.batch_estimate(pred_flow_esti_tensor, flow_network)

        # if you want to use flownet2SD, comment out the part in front
        # pred_flow_esti_tensor = torch.cat([input_last.view(-1,3,1,test_input.shape[-2],test_input.shape[-1]), G_output.view(-1,3,1,test_input.shape[-2],test_input.shape[-1])], 2)
        # gt_flow_esti_tensor = torch.cat([input_last.view(-1,3,1,test_input.shape[-2],test_input.shape[-1]), target.view(-1,3,1,test_input.shape[-2],test_input.shape[-1])], 2)
        #
        # flow_gt=flow_network(gt_flow_esti_tensor*255.0)
        # flow_pred=flow_network(pred_flow_esti_tensor*255.0)

        g_adv_loss = adversarial_loss(discriminator(G_output))
        g_op_loss = op_loss(flow_pred, flow_gt)
        g_int_loss = int_loss(G_output, target)
        g_gd_loss = gd_loss(G_output, target)

        g_loss = lam_adv * g_adv_loss + lam_gd * g_gd_loss + lam_op * g_op_loss + lam_int * g_int_loss

        optimizer_G.zero_grad()

        g_loss.backward()
        optimizer_G.step()

        train_psnr = psnr_error(G_output, target)

        # ----------- update optim_D -------
        optimizer_D.zero_grad()

        d_loss = discriminate_loss(discriminator(target), discriminator(G_output.detach()))
        # d_loss.requires_grad=True

        d_loss.backward()
        optimizer_D.step()

        # ----------- cal psnr --------------
        test_generator = generator.eval()
        test_output = test_generator(test_input)
        test_psnr = psnr_error(test_output, test_target).cuda()

        if step % 10 == 0:
            print("[{}/{}]: g_loss: {} d_loss {}".format(step, epoch, g_loss, d_loss))
            print('\t gd_loss {}, op_loss {}, int_loss {} ,'.format(g_gd_loss, g_op_loss, g_int_loss))
            print('\t train psnr{}，test_psnr {}'.format(train_psnr, test_psnr))

            writer.add_scalar('psnr/train_psnr', train_psnr, global_step=step)
            writer.add_scalar('psnr/test_psnr', test_psnr, global_step=step)

            writer.add_scalar('total_loss/g_loss', g_loss, global_step=step)
            writer.add_scalar('total_loss/d_loss', d_loss, global_step=step)
            writer.add_scalar('g_loss/adv_loss', g_adv_loss, global_step=step)
            writer.add_scalar('g_loss/op_loss', g_op_loss, global_step=step)
            writer.add_scalar('g_loss/int_loss', g_int_loss, global_step=step)
            writer.add_scalar('g_loss/gd_loss', g_gd_loss, global_step=step)

            writer.add_image('image/train_target', target[0], global_step=step)
            writer.add_image('image/train_output', G_output[0], global_step=step)
            writer.add_image('image/test_target', test_target[0], global_step=step)
            writer.add_image('image/test_output', test_output[0], global_step=step)

        step += 1

        if step % 500 == 0:
            utils.saver(generator.state_dict(), generator_model, step, max_to_save=10)
            utils.saver(discriminator.state_dict(), discriminator_model, step, max_to_save=10)
            if step >= 2000:
                print(
                    '==== begin evaluate the model of {} ===='.format(generator_model + '-' + str(step)))

                auc = evaluate(frame_num=5, layer_nums=4, input_channels=12, output_channels=3,
                               model_path=generator_model + '-' + str(step), evaluate_name='compute_auc')
                writer.add_scalar('results/auc', auc, global_step=step)

# if __name__ == '__main__':
#     train(num_clips, num_unet_layers, num_channels * (num_clips - num_his), num_channels, discriminator_channels)

# pretrain=True,
# generator_pretrain_path='../pth_model/ano_pred_avenue_generator_2.pth-4500',
# discriminator_pretrain_path='../pth_model/ano_pred_avenue_discriminator_2.pth-4500')

# test(num_clips,num_unet_layers,num_channels*(num_clips-num_his),num_channels,discriminator_channels)
# _test()
# test(0,0,0,0,0)
