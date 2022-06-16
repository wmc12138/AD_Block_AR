from models.rgb_AD_resnet import Bottleneck
from models import *
import os
import torch
from tensorboardX import SummaryWriter
params = dict()
params['gpu'] = [6,7]

writer = SummaryWriter(log_dir='./params')
model = rgb_AD_origin_resnet50(pretrained=False,num_classes=200).cuda(params['gpu'][0])
model = torch.nn.DataParallel(model, device_ids=params['gpu'])  # multi-Gpu
pretrained_dict = torch.load('./checkpoints/Uc_randn_61.513_kinetics.pth')      

model.load_state_dict(pretrained_dict['state_dict'])
state_dict = model.state_dict()
# bottleneck_layers_1 = 0
for keys in state_dict.keys():
    if 'f_weight' in keys or 'Uc' in keys:
        print(state_dict[keys])
        # writer.add_scalar('f_weight', state_dict[keys], bottleneck_layers_1)
        # bottleneck_layers_1 += 1