import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import kornia
from tensorboardX import SummaryWriter

__all__ = ['rgb_AD_origin_resnet34','rgb_AD_origin_resnet50']
# __all__ = ['ResNet', 'rgb_ADresnet18', 'rgb_ADresnet34', 'rgb_ADresnet50', 'rgb_ADresnet50_aux', 'rgb_ADresnet101',
# 'rgb_ADresnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, mode = 0):
#         super(BasicBlock, self).__init__()
#         self.mode = mode
#         self.weight_init(inplanes)
#         self.conv1_LF = nn.Conv2d(inplanes, planes*4, kernel_size=1, stride=stride, bias=False)
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def weight_init(self, inplanes):
#         self.Uc = nn.Parameter(torch.zeros(1,inplanes,1,1))
#         self.f_weight =  nn.Parameter(torch.zeros((1)))

#     def decomposer(self,x):              #x.shape  batch_size*channels*H*W
#         FL = kornia.box_blur(x,(3,3)).cuda()
#         FH = torch.sub(x, torch.mul(self.Uc, FL))
#         #FH = x - self.Uc * FL
#         # FH = x 
#         return FL,FH
    
#     def Modulator(self,LF_out, HF_out, origin_out= None):
#         assert LF_out.shape == HF_out.shape
#         out = torch.mul(HF_out,self.sigmoid(LF_out))
#         if origin_out:
#             out = torch.mul(self.f_weight,out)            #为每个ADBlock添加权重
#             out = torch.add(out,origin_out)
#         out = self.relu(out)
#         return out

#     def LF_path(self, x):
#         LF_out = self.conv1_LF(x)
#         return LF_out

#     def HF_path(self, x):
#         residual = x

#         HF_out = self.conv1(x)
#         HF_out = self.bn1(HF_out)
#         HF_out = self.relu(HF_out)

#         HF_out = self.conv2(HF_out)
#         HF_out = self.bn2(HF_out)
#         HF_out = self.relu(HF_out)

#         HF_out = self.conv3(HF_out)
#         HF_out = self.bn3(HF_out)

#         if self.downsample is not None:
#             residual = self.downsample(x)
            
#         HF_out += residual
#         # out = self.relu(out)
#         return HF_out

#     def origin_path(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         # out = self.relu(out)
#         return out

#     '''only AD forward: mode 1'''
#     def AD_forward(self, x):
#         FL,FH = self.decomposer(x)
#         LF_out = self.LF_path(FL)
#         HF_out = self.HF_path(FH)
#         out = self.Modulator(LF_out,HF_out)
#         return out

#     '''AD and ORI forward: mode 2'''
#     def AD_and_ORI_forward(self, x):
#         FL,FH = self.decomposer(x)
#         LF_out = self.LF_path(FL)
#         HF_out = self.HF_path(FH)
#         origin_out = self.origin_path(x)
#         out = self.Modulator(LF_out,HF_out,origin_out)
#         return out

#     '''ORI forward: mode 0'''
#     def ORI_forward(self, x):
#         origin_out = self.origin_path(x)
#         origin_out = self.relu(origin_out)
#         return origin_out
    
#     def forward(self,x):
#         assert self.mode == 0 or self.mode == 1 or self.mode == 2
#         if self.mode == 0:
#             out = self.ORI_forward(x)
#         elif self.mode == 1:
#             out = self.AD_forward(x)
#         elif self.mode == 2:
#             out = self.AD_and_ORI_forward(x)
#         return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, mode = 0):
        super(Bottleneck, self).__init__() 
        self.mode = mode
        self.weight_init(inplanes)
        self.conv1_LF = nn.Conv2d(inplanes, planes*4, kernel_size=1, stride=stride, bias=False)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def weight_init(self, inplanes):
        self.Uc = nn.Parameter(torch.zeros(1,inplanes,1,1))
        self.f_weight =  nn.Parameter(torch.ones((1))/2) 

    def decomposer(self,x):              #x.shape  batch_size*channels*H*W
        FL = kornia.box_blur(x,(3,3)).cuda()
        FH = torch.sub(x, torch.mul(self.Uc, FL))
        #FH = x - self.Uc * FL
        # FH = x 
        return FL,FH
    
    def Modulator(self,LF_out, HF_out, origin_out= None):
        assert LF_out.shape == HF_out.shape
        out = torch.mul(HF_out,self.sigmoid(LF_out))
        if origin_out is not None:
            out = torch.mul(self.f_weight,out)            #为每个ADBlock添加权重
            out = torch.add(out,origin_out)
        out = self.relu(out)
        return out

    def LF_path(self, x):
        LF_out = self.conv1_LF(x)
        return LF_out

    def HF_path(self, x, ori=None):
        if ori is not None:
            residual = ori
        else:
            residual = x

        HF_out = self.conv1(x)
        HF_out = self.bn1(HF_out)
        HF_out = self.relu(HF_out)

        HF_out = self.conv2(HF_out)
        HF_out = self.bn2(HF_out)
        HF_out = self.relu(HF_out)

        HF_out = self.conv3(HF_out)
        HF_out = self.bn3(HF_out)

        if self.downsample is not None:
            residual = self.downsample(residual)
            
        HF_out += residual
        # out = self.relu(out)

        return HF_out

    def origin_path(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out

    '''only AD forward: mode 1'''
    def AD_forward(self, x):
        FL,FH = self.decomposer(x)
        LF_out = self.LF_path(FL)
        HF_out = self.HF_path(FH)
        out = self.Modulator(LF_out,HF_out)
        return out

    '''AD and ORI forward: mode 2'''
    def AD_and_ORI_forward(self, x):
        FL,FH = self.decomposer(x)
        LF_out = self.LF_path(FL)
        HF_out = self.HF_path(FH)
        origin_out = self.origin_path(x)
        out = self.Modulator(LF_out,HF_out,origin_out)
        return out

    '''ORI forward: mode 0'''
    def ORI_forward(self, x):
        origin_out = self.origin_path(x)
        origin_out = self.relu(origin_out)
        return origin_out
    
    '''mode 3'''
    def mode3_forward(self,x):
        FL,FH = self.decomposer(x)
        LF_out = self.LF_path(FL)
        HF_out = self.HF_path(FH, ori=x)
        out = self.Modulator(LF_out,HF_out)
        return out

    def forward(self,x):
        assert self.mode == 0 or self.mode == 1 or self.mode == 2 or self.mode == 3
        if self.mode == 0:
            out = self.ORI_forward(x)
        elif self.mode == 1:
            out = self.AD_forward(x)
        elif self.mode == 2:
            out = self.AD_and_ORI_forward(x)
        elif self.mode == 3:
            out = self.mode3_forward(x)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=200, dropout=0.8):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        # self.fc_aux = nn.Linear(512 * block.expansion, 101)
        self.dp = nn.Dropout(p=dropout)
        self.fc_action = nn.Linear(512 * block.expansion, num_classes)
        # self.bn_final = nn.BatchNorm1d(num_classes)
        # self.fc2 = nn.Linear(num_classes, num_classes)
        # self.fc_final = nn.Linear(num_classes, 101)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if planes < 100:
            layers.append(block(self.inplanes, planes, stride, downsample, mode=0))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, mode=0))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, mode=0))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
               layers.append(block(self.inplanes, planes, mode=0))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_action(x)
        # x = self.bn_final(x)
        # x = self.fc2(x)
        # x = self.fc_final(x)

        return x


# def rgb_ADresnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
#     return model


def rgb_AD_origin_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def rgb_AD_origin_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])

        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model

# def rgb_ADresnet50_aux(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#         pretrained_dict = model_zoo.load_url(model_urls['resnet50'])

#         model_dict = model.state_dict()
#         fc_origin_weight = pretrained_dict["fc.weight"].data.numpy()
#         fc_origin_bias = pretrained_dict["fc.bias"].data.numpy()

#         # 1. filter out unnecessary keys
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         # 2. overwrite entries in the existing state dict
#         model_dict.update(pretrained_dict) 
#         # print(model_dict)
#         fc_new_weight = model_dict["fc_aux.weight"].numpy() 
#         fc_new_bias = model_dict["fc_aux.bias"].numpy() 

#         fc_new_weight[:1000, :] = fc_origin_weight
#         fc_new_bias[:1000] = fc_origin_bias

#         model_dict["fc_aux.weight"] = torch.from_numpy(fc_new_weight)
#         model_dict["fc_aux.bias"] = torch.from_numpy(fc_new_bias)

#         # 3. load the new state dict
#         model.load_state_dict(model_dict)

#     return model

# def rgb_ADresnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
#     return model


# def rgb_ADresnet152(pretrained=False, **kwargs):
#     """Constructs a ResNet-152 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     if pretrained:
#         # model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
#         pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
#         model_dict = model.state_dict()

#         # 1. filter out unnecessary keys
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         # 2. overwrite entries in the existing state dict
#         model_dict.update(pretrained_dict) 
#         # 3. load the new state dict
#         model.load_state_dict(model_dict)

#     return model


if __name__ == "__main__":
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="6"
    # model = rgb_resnet152(pretrained=True, num_classes=101)
    # state_dict = model.state_dict()
    # # print(model)
    # # print(state_dict.keys())
    # print(state_dict['conv1.weight'].size())
    model = rgb_AD_origin_resnet50(pretrained=True,num_classes=101).cuda()
    state_dict = model.state_dict()
    # print(model)
    # print(state_dict.keys())
    # out = model(fake_img)
    # print(out)