import numpy as np
from tensorboardX import SummaryWriter
import json
import os

'''测试集仍然使用的验证集'''
root = "/home/WangMaochuan/datasets/kinetics_400/val_jpgs"
validate_json = "/home/WangMaochuan/two-stream-ADBlock/kinetics_utils/validate.json"
validate_list = "/home/WangMaochuan/two-stream-ADBlock/kinetics_utils/mini_kinetics_200_val.txt"
label_json = "/home/WangMaochuan/two-stream-ADBlock/kinetics_utils/label.json"
temporal_log_path = 'temporal_log.txt'
test_list_path = '/home/WangMaochuan/two-stream-ADBlock/scripts/eval_minikinetics_pytorch/spatial_testlist_with_labels.txt'

def gen_mini_kinetics_testlist_with_labels():
    
    with open(validate_json,'r') as file:
            json_dict = json.load(file)

    with open(label_json, 'r') as file:
            label_dict = json.load(file)

    with open (validate_list, 'r') as file:
            test_list = file.readlines()

    with open(test_list_path, 'w') as file:
        for line in test_list:
            id = line.split()[0]
            if id in json_dict:
                label = json_dict[id]['annotations']['label']
                label_id = label_dict[label]
                clip_path = os.path.join(root,label+'_'+id)
                if os.path.exists(clip_path):
                    file.write(str(label_id) + ' ' + clip_path + '\n')

def gen_class_dict(path):

    with open(path, 'r') as file:
        label_dict = json.load(file)
    class_dict = {}
    for key,value in label_dict.items():
        class_dict[value] = key
    return class_dict

class my_log:

    def __init__(self,path):
        self.path = path
        with open(self.path,'a') as file:
            file.write('\n\n\nnew:')
            file.write('-'*100)

    def write(self,info):
        with open(self.path,'a') as file:
            file.write('\n'+info)

# def tensorboard_log_gen(mode,file_path,start_line,end_line):
#     if mode == 'rgb':
#         logdir = '../../log/saptial'
#     else:
#         logdir = '../../log/temporal'
#     # writer = SummaryWriter(log_dir=logdir)
#     with open(file_path,'r',encoding='gbk') as file:
#         lines = file.readlines()
#     epoch = 0
#     for i in range(end_line-start_line+1):
#         if lines[start_line+i][8] == str(epoch):
#             writer.add_scalar('train_loss_epoch', losses.avg, epoch)
#             writer.add_scalar('train_top1_acc_epoch', top1.avg, epoch)
#             writer.add_scalar('train_top3_acc_epoch', top3.avg, epoch)
#             epoch += 1


#     # writer.close


def topK_accuracy(pred,input_video_label,topK=(1,3)):
    res = [False,False]
    pred_index_sort = np.argsort(-pred)
    for i in range(len(topK)):
        for j in range(topK[i]):
            if pred_index_sort[j] == input_video_label:
                res[i] = True
    return res
# def topK_accuracy(pred,input_video_label,topK=(1,3)):
#     res = [False,False]
#     pred_dict = {str(pred[i]):i for i in range(len(pred))}
#     pred.sort()
#     for i in range(len(topK)):
#         for j in range(topK[i]):
#             if pred_dict[str(pred[-1-j])]== input_video_label:
#                 res[i] = True
#     return res

if __name__ == '__main__':
    pass
    # gen_mini_kinetics_testlist_with_labels()
    # print(gen_class_dict("/home/WangMaochuan/two-stream-ADBlock/kinetics_utils/label.json"))
