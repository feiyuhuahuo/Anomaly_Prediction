import torch
import glob
import os

def saver(model_state_dict,model_path,step,max_to_save=5):
    total_models=glob.glob(model_path+'*')
    if len(total_models)>=max_to_save:
        total_models.sort()
        os.remove(total_models[0])
    torch.save(model_state_dict,model_path+'-'+str(step))
    print('model {} save successfully!'.format(model_path+'-'+str(step)))
