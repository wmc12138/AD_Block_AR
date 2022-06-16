import numpy as np
import cv2
import os
import sys
import json

def generate_jpgs(video_file, target_dir, label):
    cap = cv2.VideoCapture(video_file)  # 获取到一个视频
    isOpened = cap.isOpened()  # 判断是否打开
    # 为单张视频，以视频名称所谓文件名，创建文件夹
    id = os.path.split(video_file)[-1][:-18]
    dir_name = label + '_' + id
    single_pic_store_dir = os.path.join(target_dir, dir_name)
    if not os.path.exists(single_pic_store_dir):
        os.mkdir(single_pic_store_dir)
    i = 0
    while isOpened:
        i += 1
        (flag, frame) = cap.read()  # 读取一张图像
        fileName = 'frame' + '%06d'%i + ".jpg"
        if (flag == True):
            # 以下三行 进行 旋转
            #frame = np.rot90(frame, -1)
            #print(fileName)
            # 设置保存路径
            if min(frame.shape[:2]) >= 512:
                x,y = frame.shape[:2]
                frame = cv2.resize(frame, (int(y/2), int(x/2)))
            save_path = os.path.join(single_pic_store_dir, fileName)
            #print(save_path)
            cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        else:
            break
    print('%s视频已成功转换为帧保持至文件夹：“%s”'%(id,single_pic_store_dir))

def kinetics_jpgs_generate(videos_path, json_path, target_dir):
    with open(json_path,'r') as file:
        json_dict = json.load(file)
    path_dict = os.listdir(videos_path)

    for path in path_dict:
        if path[-3:] == 'mp4':
            video_path = os.path.join(videos_path,path)
            key = path[:-18]
            if key in json_dict:
                label = json_dict[key]['annotations']['label']
                generate_jpgs(video_path, target_dir, label)
    
if __name__ == '__main__':
    kinetics_jpgs_generate('/home/WangMaochuan/datasets/kinetics-400/test_videos', '/home/WangMaochuan/two-stream-ADBlock/kinetics_utils/test.json', '/home/WangMaochuan/datasets/kinetics-400/test_jpgs')
    # generate_jpgs('./video.avi','./')
    # print(os.path.split('asd/adsa/--asdas5456_asda.avi'))
    # a=os.listdir('/home/WangMaochuan/datasets/kinetics-400/val_videos')
    # print(len(a))
    # print(a[155][-4:])
    # print(a[155][:-18])
    # key = a[155][:-18]
    # with open('/home/WangMaochuan/two-stream-ADBlock/kinetics_utils/validate.json','r') as file:
    #     json_dict = json.load(file)
    # if key in json_dict:
    #     print(json_dict[key]['annotations']['label'])
    # print(json_dict[key]['annotations']['label'])
    # for i,path in enumerate(a):
    #     print(i,':',path)

