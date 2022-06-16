import json
import pandas as pd

def convert_kinetics_csv_to_json(csv_path):
    data = pd.read_csv(csv_path)
    keys = []
    key_labels = []
    segments = []
    # for i in range(10):
    for i in range(data.shape[0]):
        row = data.iloc[i,:]
        basename = '%s'%(row['youtube_id'])
        keys.append(basename)
        key_labels.append(row['label'])
        segments.append([float(row['time_start']),float(row['time_end'])])
    
    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = row['split']
        database[key]['annotations'] = {'label':key_labels[i],'segment':segments[i]}

    with open('%s.json'%(row['split']),'w') as dst_file:
        json.dump(database, dst_file, indent=4, sort_keys=True)

def convert_csv_to_dict(csv_path):
    data = pd.read_csv(csv_path)
    keys = []
    key_labels = []
    segments = []
    # for i in range(10):
    for i in range(data.shape[0]):
        row = data.iloc[i,:]
        basename = '%s'%(row['youtube_id'])
        keys.append(basename)
        key_labels.append(row['label'])
        segments.append([float(row['time_start']),float(row['time_end'])])
    
    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = row['split']
        database[key]['annotations'] = {'label':key_labels[i],'segment':segments[i]}
    return database

def load_labels(train_csv_path, total_train_json_path):
    with open(train_csv_path, 'r') as file:
        id_list = file.readlines()
    with open(total_train_json_path, 'r') as file:
        json_dict = json.load(file)
    label_list = []
    for id in id_list:
        id = id.split()[0]
        if id in json_dict:
            label = json_dict[id]['annotations']['label']
            label_list.append(label)
    unique_label_list = sorted(list(set(label_list)), key=str.lower)
    label_dict = {}
    for i,label in enumerate(unique_label_list):
        label_dict[label] = i
    with open('label.json','w') as dst_file:
        json.dump(label_dict, dst_file, indent=4, sort_keys=True)

def convert_all_kinetics_csv_to_json(train_csv_path, test_csv_path, validate_csv_path):
    labels = load_labels(train_csv_path)
    train_database = convert_csv_to_dict(train_csv_path)
    test_database = convert_csv_to_dict(test_csv_path)
    validate_database = convert_csv_to_dict(validate_csv_path)

    dst_data = {}
    dst_data['labels'] = labels
    # dst_data['train_database']={}
    # dst_data['train_database'].update(train_database)
    # dst_data['test_database']={}
    # dst_data['test_database'].update(test_database)
    # dst_data['validate_database']={}
    # dst_data['validate_database'].update(validate_database)

    # with open('all_database.json','w') as dst_file:
    #     json.dump(dst_data, dst_file, indent=4, sort_keys=True)

    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(test_database)
    dst_data['database'].update(validate_database)


if __name__ == '__main__':
    # convert_kinetics_csv_to_json('./train.csv')
    # convert_kinetics_csv_to_json('./test.csv')
    # convert_kinetics_csv_to_json('./validate.csv')
    # convert_all_kinetics_csv_to_json('./train.csv','./test.csv','./validate.csv')
    # print('video.avi'.suffix)
    # with open('./validate.json','r') as file:
    #     database = json.load(file)
    # print(type(database))
    # print(database['--uGS0Y4D6k']['annotations']['label'])
    # for k,v in database.items():
    #     print(k,v)

    # a={'ss':1,'sadf':2,'sada':'sad'}
    # for k,v in a.items():
    #     print(k,v)
    # load_labels('mini_kinetics_200_train.txt','train.json')
    # with open('label.json','r') as file:
    #     a = json.load(file)
    # for label,num in a.items():
    #     print(label,num,type(num))
    pass
