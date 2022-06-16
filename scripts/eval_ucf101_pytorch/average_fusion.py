#对空间流和时间流预测输出的(3783,101)数组进行按权重融合并重新预测标签
#TODO 我已将测试集某个测试不了的视频从测试集中删除，原序号为1806
import utils
import numpy as np

ucf101_spatial_pred_array_path  = './AD_origin_lr0.001_dropout0.7.npy'
ucf101_temporal_pred_array_path = './ucf_flow_resnet50_model_best_split1.npy'
ucf101_test_label_path          = './my_spatial_testlist01_with_labels.txt'
spatial_weight = 1
temporal_weight = 2
log_path = 'average_fusion_log.txt'

def diff_stream_res(labels,pred_array,stream_name,log):
    match_count_top1 = 0
    match_count_top3 = 0

    for i in range(len(labels)):
        res = utils.topK_accuracy(pred_array[i],labels[i],topK=(1,3))
        if res[0]:
            match_count_top1 += 1
            match_count_top3 += 1
        elif res[1]:
            match_count_top3 += 1
        else:
            pass
    info = '{}结果如下：\n预测集视频总数为：{}\ntop1预测正确数量：{}\ntop3预测正确数量：{}\n@top1_accuracy:{}\n@top3_accuracy:{}\n'\
        .format(stream_name,len(labels),match_count_top1,match_count_top3,match_count_top1 / len(labels),match_count_top3 / len(labels))
    print(info)
    log.write(info)

def main(spatial_pred_path,temporal_pred_path,label_path,spatial_weight,temporal_weight,log_path):
    with open(label_path,'r') as label_file:
        content = label_file.readlines()
        labels = [int(line.split()[-1])-1 for line in content]

    spatial_pred_array = np.load(spatial_pred_path)
    if spatial_pred_array.shape[0] == 3783:            #这里是我最初测试空间流时测试集全用上了，测试时间流时减少了第1806个视频。
        spatial_pred_array = np.delete(spatial_pred_array,1805,axis=0)
    temporal_pred_array = np.load(temporal_pred_path)
    fusion_pred_array = spatial_weight*spatial_pred_array + temporal_weight*temporal_pred_array

    log = utils.my_log(log_path)
    info = '本次fusion:\nlabel路径为:{}\n spatial数据路径为:{}\ntemporal数据路径为:{}\n spitial权重为：{}\ntemporal权重为：{}\n'\
        .format(label_path,spatial_pred_path,temporal_pred_path,spatial_weight,temporal_weight)
    log.write(info)

    diff_stream_res(labels, spatial_pred_array, 'spatial_stream_net',log)
    diff_stream_res(labels, temporal_pred_array, 'temporal_stream_net',log)
    diff_stream_res(labels, fusion_pred_array, 'fusion_two_streams_net',log)


if __name__ == '__main__':
    main(ucf101_spatial_pred_array_path,ucf101_temporal_pred_array_path,ucf101_test_label_path,spatial_weight,temporal_weight,log_path)

