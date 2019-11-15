import get_flow
from models import FlowNet2SD
from networks import FlowNetSD
import numpy as np
import torch
from liteFlownet.flow_vis import vis_mv
from scipy.misc import imread,imresize
import argparse
import os
from Dataset.img_dataset import np_load_frame
# img1='../liteFlownet/001.jpg'
# img2='../liteFlownet/002.jpg'
# img1='/hdd/fjc/VAD/ped1/training/frames/ped1_train_01/005.jpg'
# img2='/hdd/fjc/VAD/ped1/training/frames/ped1_train_01/007.jpg'
# img1='/hdd/fjc/VAD/ped2/training/frames/01/006.jpg'
# img2='/hdd/fjc/VAD/ped2/training/frames/01/007.jpg'
# img1='/hdd/fjc/VAD/shanghaitech/training/frames/01/01_001/01_001_0069.jpg'
# img2='/hdd/fjc/VAD/shanghaitech/training/frames/01/01_001/01_001_0072.jpg'
img1='/hdd/fjc/VAD/shanghaitech/training/frames/01_002/01_002_0075.jpg'
img2='/hdd/fjc/VAD/shanghaitech/training/frames/01_002/01_002_0076.jpg'

# img1='/hdd/fjc/VAD/avenue/training/frames/01/0003.jpg'
# img2='/hdd/fjc/VAD/avenue/training/frames/01/0004.jpg'

model_path='/home/fjc/FlowNet2-SD_checkpoint.pth.tar'
flowSD_model_path='/home/fjc/FlowNetSD/flownet-SD.pth'

os.environ['CUDA_VISIBLE_DEVICES']='3'

def test():
    jpg1=imread(img1)
    jpg2=imread(img2)
    #[3,256,256]
    jpg1=imresize(jpg1,(384,512))
    jpg2=imresize(jpg2,(384,512))

    # jpg1=np.expand_dims(jpg1,1)
    # jpg2=np.expand_dims(jpg2,1)
    #
    # #[3,2,256,256]
    # images=np.concatenate([jpg1,jpg2],1)
    images=np.array([jpg1,jpg2],np.float32)
    #[2,256,256,3]
    # images=[jpg1,jpg2]
    #
    # images=np.array(images)

    images=np.transpose(images,[3,0,1,2])


    images=torch.FloatTensor(images).cuda()

    # flownet=FlowNetSD.FlowNetSD(False).cuda().eval()
    #
    # flownet.load_state_dict(torch.load(flowSD_model_path))

    flownet=FlowNet2SD().cuda().eval()
    flownet.load_state_dict(torch.load(model_path)['state_dict'])

    flow=get_flow.get_flow(flownet,images)
    flow=flow.cpu().detach().numpy()
    vis_mv(flow[0].transpose([1,2,0]))
    print(flow[0].transpose(1,2,0))

def test_flownet():
    flownetSD=FlowNetSD.FlowNetSD().cuda().eval()
    a=flownetSD.state_dict()
    print(a)

if __name__=='__main__':
    test()


