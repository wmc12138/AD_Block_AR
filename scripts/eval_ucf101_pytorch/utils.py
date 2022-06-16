import numpy as np
from tensorboardX import SummaryWriter

my_jpeg_frame_path = '../../../datasets/UCF101_jpegs_256/'
my_flow_frame_path = '../../../datasets/UCF101_tvl1_flow/'
class_ind_path = '../../datasets/ucf101_splits/classInd.txt'
spatial_log_path = 'spatial_log.txt'
temporal_log_path = 'temporal_log.txt'

def jpeg_path_change(path):
    f1 = open('spatial_testlist01_with_labels.txt','r')
    val_lists = f1.readlines()
    f2 = open('my_spatial_testlist01_with_labels.txt','w')
    for line in val_lists:
        f2.write(path+line.split('/')[-1])
    f1.close()
    f2.close()

def flow_path_change(path):
    f1 = open('temporal_testlist01_with_labels.txt','r')
    val_lists = f1.read().split('\n')
    f2 = open('my_temporal_testlist01_with_labels.txt','w')
    for line in val_lists[:-1]:
        f2.write(path+'u/'+line.split('/')[-1].split()[0] +' '+path+'v/'+line.split('/')[-1]+'\n')
    f1.close()
    f2.close()

def gen_class_dict(path) :
    class_dict = {}
    with open(path) as class_file:
        lists = class_file.readlines()
        for line in lists:
            class_dict[int(line.split()[0])-1] = line.split()[1]
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
    tensorboard_log_gen('rgb',file_path='../../train_spatial_log.txt',start_line=10,end_line=20)
    # jpeg_path_change(my_jpeg_frame_path)
    # flow_path_change(my_flow_frame_path)
    # gen_class_dict(class_ind_path)