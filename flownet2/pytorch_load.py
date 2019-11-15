import h5py
import torch
from networks import FlowNetSD
import tensorflow as tf
from models.unet import Unet
# def load_conv2d(state_dict,name_pth,name_tf):
def tf_model_pth(checkpoint_path,pth_output_path):
    model=FlowNetSD.FlowNetSD(batchNorm=False).eval()
    state_dict=model.state_dict()

    with open(checkpoint_path) as f:
        ckptFileName=f.readline().split('"')[1]

    reader=tf.train.NewCheckpointReader(ckptFileName)

    pth_keys=state_dict.keys()
    keys=sorted(reader.get_variable_to_shape_map().keys())
    for pth_key in pth_keys:
        pth_keySplits=pth_key.split('.')
        key_pre='FlowNetSD/'

        if pth_keySplits[0][:8]=='upsample':
            key=key_pre+'upsample_flow'+pth_keySplits[0][-6]+'to'+pth_keySplits[0][-1]+'/'+pth_keySplits[-1]
        elif pth_keySplits[0][:5]=='inter':
            key=key_pre+'interconv'+pth_keySplits[0][-1]+'/'+pth_keySplits[-1]
        else:
            key=key_pre+pth_keySplits[0]+'/'+pth_keySplits[-1]

        if pth_keySplits[-1]=='weight':
            tensor = reader.get_tensor(key+'s')
            state_dict[pth_key]=torch.from_numpy(tensor).permute([3,2,0,1])
        else:
            tensor = reader.get_tensor(key+'es')
            state_dict[pth_key]=torch.from_numpy(tensor)

    torch.save(model.state_dict(),pth_output_path)

def unet_tf_pth(checkpoint_path,pth_output_path):
    model=Unet().eval()
    state_dict=model.state_dict()

    reader=tf.train.NewCheckpointReader(checkpoint_path)

    pth_keys=state_dict.keys()
    keys=sorted(reader.get_variable_to_shape_map().keys())
    print(keys)
    print(pth_keys)
    # for pth_key in pth_keys:


if __name__=='__main__':
    #tf_model_pth(r'/home/fjc/FlowNetSD/checkpoint',r'/home/fjc/FlowNetSD/flownet-SD.pth')
    unet_tf_pth(r'/home/fjc/pretrains/ped1',r'/home/fjc/trans_pth/ped1.pth')
