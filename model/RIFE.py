import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *
from model.LSTM_attention import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
'''
import yaml
with open('/home/jyzhao/Code/ResLSTM/CustomConfig.yaml",','r') as file:
    config = yaml.safe_load(file)
'''

class Model:
    def __init__(self, pretrainedModel, local_rank=-1, arbitrary=False):

        self.flownet = VSRbackbone(pretrainedModel)
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.lap = LapLoss()
        self.sobel = SOBEL()
        # if local_rank != -1:
        #    self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()
        
    def parameter_loaded_train(self):
        # Load parameters for Ori_IFnet
        self.device()
        self.flownet.train()
    
    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)
        
    def simple_inference(self, imgs):
        # print('enter')
        self.flownet.eval()
        with torch.no_grad():
            # Prepare the inputs

            # Forward pass through the model
            # merged = self.flownet(imgs)
            if False:
                _, _, output_allframes, _, _, _, _, _,_= self.flownet(imgs, training_flag=False)
            if True:
                _, _, output_allframes, _, _, _, _, _, _= self.flownet(imgs, training_flag=False)

            # print('mergedout')
            # Assuming merged is the output tensor containing the interpolated frame
            return output_allframes.squeeze(0)  # Remove the batch dimension

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            self.flownet.load_state_dict(torch.load('{}/flownet.pkl'.format(path)))
           
    def save_model(self, path, rank=0):
        if not os.path.exists(path):
            os.makedirs(path)
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))


    def update(self, allframes, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate

        if training:
            self.train()
        else:
            self.eval()
        flow_list, mask_list, output_allframes, flow_teacher_list, output_teacher, Sum_loss_tea_pred, Sum_loss_context, Sum_loss_mse,loss_dist = self.flownet(allframes)

        if training:
            torch.autograd.set_detect_anomaly(True)
            self.optimG.zero_grad()
            loss_G = Sum_loss_tea_pred + Sum_loss_context + 0.01*loss_dist
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.flownet.parameters(), max_norm=1.0)
            self.optimG.step()
        else:
            pass
            # flow_teacher = flow[2]
        return output_allframes, output_teacher,{
            'merged_tea': output_teacher[-1],
            'mask': mask_list[-1],
            #'mask_tea': mask,
            'flow': flow_list[-1],
            'flow_tea': flow_teacher_list[-1],
            'Sum_loss_context': Sum_loss_context,
            'Sum_loss_mse': Sum_loss_mse,
            'Sum_loss_tea_pred': Sum_loss_tea_pred,
            "loss_dist":loss_dist
            }
