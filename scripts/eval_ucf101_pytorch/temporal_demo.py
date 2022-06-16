#!/usr/bin/env python

import os
import sys
import collections
import numpy as np
import cv2
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.insert(0, "../../")
import models
import utils
from VideoTemporalPrediction import VideoTemporalPrediction

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
param = dict()
param['gpu'] = [0,1,2,3]
temporal_log_path = 'temporal_log.txt'

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z


def main():

    model_path = '../../checkpoints/flow_resnet50_model_best_split1.pth'
    data_dir = "../../../datasets/UCF101_tvl1_flow"
    start_frame = 0
    num_categories = 101

    model_start_time = time.time()
    params = torch.load(model_path)
    temporal_net = models.flow_resnet50(pretrained=False, num_classes=101)
    temporal_net = temporal_net.cuda(param['gpu'][0])
    temporal_net = nn.DataParallel(temporal_net, device_ids=param['gpu'])  # multi-Gpu
    temporal_net.load_state_dict(params['state_dict'])
    temporal_net.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition temporal model is loaded in %4.4f seconds." % (model_time))

    val_file = "./my_temporal_testlist01_with_labels.txt"
    f_val = open(val_file, "r")
    val_list = f_val.readlines()
    print("we got %d test videos" % len(val_list))

    line_id = 1
    match_count_top1 = 0
    match_count_top3 = 0
    result_list = []
    class_dict = utils.gen_class_dict('../../datasets/ucf101_splits/classInd.txt')
    temporal_log = utils.my_log(temporal_log_path)
    temporal_log.write('本次测试采用的训练好的网络为：%s' % model_path.split('/')[-1])
    info = '    视频序号/总数      视频行为            预测行为       视频标签       预测标签    top1    top3'
    temporal_log.write(info)
    print('视频序号/总数      视频行为            预测行为      视频标签    预测标签    top1    top3')

    for line in val_list:
        line_info = line.split(" ")
        clip_path = (line_info[0],line_info[1])
        input_video_label = int(line_info[2]) - 1

        spatial_prediction = VideoTemporalPrediction(
                clip_path,
                temporal_net,
                num_categories,
                start_frame)

        avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
        # print(avg_spatial_pred_fc8.shape)
        result_list.append(avg_spatial_pred_fc8)
        # avg_spatial_pred = softmax(avg_spatial_pred_fc8)

        pred_index = np.argmax(avg_spatial_pred_fc8)
        res = utils.topK_accuracy(avg_spatial_pred_fc8, input_video_label, topK=(1, 3))
        info="     {:>2}/{:<4}    {:<20}{:<20}{:<3}          {:<3}     {:<5}   {:<5}".format(line_id, len(val_list), class_dict[input_video_label], class_dict[pred_index], input_video_label, pred_index,str(res[0]),str(res[1]))
        print(info)
        temporal_log.write(info)
        if res[0]:
            match_count_top1 += 1
            match_count_top3 += 1
        elif res[1]:
            match_count_top3 += 1
        else:
            pass
        line_id += 1
    info = 'top1正确预测视频行为数量：%d\ntop3正确预测视频行为数量：%d\n视频总数量：%d\ntop1_Accuracy is %4.4f\ntop3_Accuracy is %4.4f'%(
    match_count_top1,match_count_top3,len(val_list),float(match_count_top1) / len(val_list),
    float(match_count_top3) / len(val_list))
    print(info)
    temporal_log.write(info)
    np.save("ucf_%s" % model_path.split('/')[-1].split('.')[0], np.array(result_list))

if __name__ == "__main__":
    main()




    # # spatial net prediction
    # class_list = os.listdir(data_dir)
    # class_list.sort()
    # print(class_list)

    # class_index = 0
    # match_count = 0
    # total_clip = 1
    # result_list = []

    # for each_class in class_list:
    #     class_path = os.path.join(data_dir, each_class)

    #     clip_list = os.listdir(class_path)
    #     clip_list.sort()

    #     for each_clip in clip_list:
            # clip_path = os.path.join(class_path, each_clip)
            # spatial_prediction = VideoSpatialPrediction(
            #         clip_path,
            #         spatial_net,
            #         num_categories,
            #         start_frame)

            # avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
            # # print(avg_spatial_pred_fc8.shape)
            # result_list.append(avg_spatial_pred_fc8)
            # # avg_spatial_pred = softmax(avg_spatial_pred_fc8)

            # pred_index = np.argmax(avg_spatial_pred_fc8)
            # print("GT: %d, Prediction: %d" % (class_index, pred_index))

            # if pred_index == class_index:
            #     match_count += 1
#             total_clip += 1

#         class_index += 1

#     print("Accuracy is %4.4f" % (float(match_count)/total_clip))
#     np.save("ucf101_split1_resnet_rgb.npy", np.array(result_list))

# if __name__ == "__main__":
#     main()
