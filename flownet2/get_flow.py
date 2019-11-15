from models import *
import torch
# from networks import *
from liteFlownet.flow_vis import vis_mv

def get_flow(model,input_tensor):
    '''
    input tensor is in range [0,1] and 
    :param model: the loaded flownetSD model
    :param input_tensor: the pytorch tensor,shape as [batch_size,channels*2,width,height]
    range from[0,1]
    :return:
    '''
    # the flownet2 need
    #to the scale of [0,255]
    # input_tensor=input_tensor*255.0
    input_tensor=input_tensor.view([-1,3,2,input_tensor.shape[-2],input_tensor.shape[-1]])

    model.eval()
    flow=model(input_tensor)

    return flow

def get_batch_flow(model,input_tensor):
    #[batch_size,channels*2,height,weight]
    input_tensor=input_tensor*255.0
    # input_tensor=input_tensor.view([input_tensor[0],-1,2,input_tensor.shape[-2],input_tensor.shape[-1]])
    flow=model(input_tensor)
    return flow


