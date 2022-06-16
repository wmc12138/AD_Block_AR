import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2
import json

# def find_classes(dir):
#     classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#     classes.sort()
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     return classes, class_to_idx

def make_dataset(root, source, phase, label_json="/home/WangMaochuan/two-stream-ADBlock/kinetics_utils/label.json", modality=None):

    if not os.path.exists(source):
        print("Setting file %s doesn't exist." % (source))
        sys.exit()
    else:
        clips = []
        if phase == 'train':
            with open("/home/WangMaochuan/two-stream-ADBlock/kinetics_utils/train.json",'r') as file:
                json_dict = json.load(file)
                root_path = os.path.join(root, 'train_jpgs')
        elif phase == 'val':
            with open("/home/WangMaochuan/two-stream-ADBlock/kinetics_utils/validate.json",'r') as file:
                json_dict = json.load(file)
                root_path = os.path.join(root, 'val_jpgs')
        else: 
            print('wrong train or validate file path!')

        with open(label_json, 'r') as file:
            label_dict = json.load(file)

        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                id = line.split()[0]
                if id in json_dict:
                    label = json_dict[id]['annotations']['label']
                    target = label_dict[label]
                    clip_path = os.path.join(root_path, label+ '_'+ id)
                    if os.path.exists(clip_path):
                        duration = len(os.listdir(clip_path))
                        if duration != 0:
                            item = (clip_path, duration, target)
                            clips.append(item)
    return clips

def ReadSegmentRGB(path, offset, new_height, new_width, new_length, is_color, name_pattern):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []
    for length_id in range(1, new_length+1):
        frame_name = name_pattern % (length_id + offset)
        frame_path = path + "/" + frame_name
        cv_img_origin = cv2.imread(frame_path, cv_read_flag)
        if cv_img_origin is None:
            print("Could not load file %s" % (frame_path))
            sys.exit()
            # TODO: error handling here
        if new_width > 0 and new_height > 0:
            # use OpenCV3, use OpenCV2.4.13 may have error
            cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
        else:
            cv_img = cv_img_origin
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        sampled_list.append(cv_img)
    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input

class mini_kinetics(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 modality,
                 name_pattern=None,
                 is_color=True,
                 new_length=1,
                 new_width=0,
                 new_height=0,
                 transform=None,
                 target_transform=None,
                 video_transform=None):

        # classes, class_to_idx = find_classes(root)
        clips = make_dataset(root, source, phase)

        if len(clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory."))   #有问题，应该是txt文件无内容才对

        self.root = root
        self.source = source
        self.phase = phase
        self.modality = modality

        # self.classes = classes
        # self.class_to_idx = class_to_idx
        self.clips = clips

        if name_pattern:
            self.name_pattern = name_pattern
        else:
            if self.modality == "rgb":
                self.name_pattern = "frame%06d.jpg"
            elif self.modality == "flow":
                self.name_pattern = "frame%06d.jpg"

        self.is_color = is_color
        self.new_length = new_length
        self.new_width = new_width
        self.new_height = new_height

        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

    def __getitem__(self, index):
        path, duration, target = self.clips[index]
        if self.phase == "train":
            if duration >= self.new_length:
                offset = random.randint(0, duration - self.new_length)
                # No +1 because randint(a,b) return a random integer N such that a <= N <= b.
            else:
                offset = 0
        elif self.phase == "val":
            if duration >= self.new_length:
                offset = int((duration - self.new_length + 1)/2)
            else:
                offset = 0
        else:
            print("Only phase train and val are supported.")


        if self.modality == "rgb":
            clip_input = ReadSegmentRGB(path,
                                        offset,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern
                                        )

        else:
            print("No such modality %s" % (self.modality))

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)

        return clip_input, target


    def __len__(self):
        return len(self.clips)

if __name__ == '__main__':
    dataset = mini_kinetics(root="/home/WangMaochuan/datasets/kinetics_400",source="/home/WangMaochuan/two-stream-ADBlock/kinetics_utils/mini_kinetics_200_train.txt",phase='train',modality='rgb')
    print(dataset)