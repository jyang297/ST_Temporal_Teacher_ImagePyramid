import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from model.LSTM_attention import *
from model.RIFE import Model
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from model.VimeoSeptuplet import *

device = torch.device("cuda")

log_path = 'train_log'
intrain_path = 'intrain_log'

from model.pretrained_RIFE_loader import IFNet_update
from model.pretrained_RIFE_loader import convert_load



if __name__ == "__main__":
    torch.cuda.empty_cache()
   
    torch.cuda.set_device(-1)

    torch.backends.cudnn.benchmark = True
    pretrained_path = 'RIFE_log'
    checkpoint = convert_load(torch.load(f'{pretrained_path}/flownet.pkl', map_location=device))
    Ori_IFNet_loaded = IFNet_update()
    Ori_IFNet_loaded.load_state_dict(checkpoint)
    for param in Ori_IFNet_loaded.parameters():
        param.requires_grad = False

    model = Model(Ori_IFNet_loaded, -1)
    # Dummy parameters
    step = 5555
    # dataset_val = VimeoDatasetSep('test')
    # val_data = DataLoader(dataset_val, batch_size=4, pin_memory=True, num_workers=1)
    # writer_val = SummaryWriter('validate')

    pretrained_model_path = '/home/jyang297/projects/def-jyzhao/jyang297/Ablation/Upload_image_pyramid/intrain_log'
    model.load_model(pretrained_model_path)
    print("Loaded ConvLSTM model")
    # model.eval()

    # evaluate(model, val_data, step, args.local_rank, writer_val)
