import os
import glob
import shutil
import sys
sys.path.append("/home/hao/Project/MERTools/MER2024")
from toolkit.globals import *
from toolkit.utils.read_files import *
from toolkit.utils.functions import *


## 解析 train_label
def read_train_label(train_video, train_label):
    video_names = os.listdir(train_video)
    
    video_names = [item.rsplit('.')[0] for item in video_names]

    name2emo = {}
    names = func_read_key_from_csv(train_label, 'name')
    emos  = func_read_key_from_csv(train_label, 'discrete')
    for (name, emo) in zip(names, emos):
        name2emo[name] = emo
    
    train_names = video_names
    train_emos = [name2emo[name] for name in train_names]
    print(f'train: {len(video_names)}')
    return train_names, train_emos


## 解析 test_label
def read_test_label(test_video, test_label):

    new_candidate_path = '/data/public_datasets/MER_2024/new_release/candidate_20000.csv'
    ## 从 candidate_20000.csv 读取候选视频名称
    candidate_df = pd.read_csv(new_candidate_path)
    candidate_names = set(candidate_df['name'].tolist())  # 假设候选文件中有一列名为 'video_name'

    video_names = os.listdir(test_video)
    video_names = [item.rsplit('.')[0] for item in video_names]

    # 过滤只包含在 candidate_names 中的视频
    filtered_video_names = [name for name in video_names if name in candidate_names]
    emos = ['neutral'] * len(filtered_video_names)
    print(f'test: {len(emos)}')
    return filtered_video_names, emos

## 解析 train_20000_label 因为源地址下面有所有的11.5w个视频，需要只读20000个
def read_train_20000_label(soft_train_video, soft_train_label):

    new_candidate_path = '/data/public_datasets/MER_2024/new_release/candidate_20000.csv'  #先选择20000个的
    ## 从 candidate_20000.csv 读取候选视频名称
    candidate_df = pd.read_csv(new_candidate_path)  # 读20000个所对应的csv
    candidate_names = set(candidate_df['name'].tolist())   # 将name列转换成list，再转换成set  

    video_names = os.listdir(soft_train_video)  # video-unlabeled-with-test2noise文件夹路径，里面存放的是所有的视频（11w+）
    video_names = [item.rsplit('.')[0] for item in video_names if item.rsplit('.')[0] in candidate_names]  # 找到在20000个中存在的伪标签对应视频名称,将范围缩小到20000
    print(len(video_names))
    print(type(video_names))

    name2emo = {}  # name to emo
    names = func_read_key_from_csv(soft_train_label, 'name')  # 获取names，提取成伪标签以后的names
    emos  = func_read_key_from_csv(soft_train_label, 'discrete')  # 获取emos（伪标签）


    for (name, emo) in zip(names, emos):  # # 按照映射关系保存为字典,伪标签候选视频名称为name,并针对2w条进行筛选
        if name in video_names:
            name2emo[name] = emo

    soft_names = list(name2emo.keys())
    soft_emos = list(name2emo.values())
    
    # train_names = video_names  # 将videonames（20000）赋值给新的train_names
    # # 发现当name中的视频名称不在train_names中时，会因为找不到键而报错，需要将找不到字典key（在11w不在2w的匹配去除）
    # train_emos = [name2emo[name] for name in train_names]  # 开始遍历查找1.for name（伪） in train_names（20000） 2.name2emo[name]=>emo-label =train_emos
    
    print(f'train_20000_match_to_softlabel: {len(soft_names)}')
    return soft_names, soft_emos


def normalize_dataset_format(data_root, save_root):

    ## input path
    train_video, train_label = os.path.join(data_root, 'video-labeled'),   os.path.join(data_root, 'label-disdim.csv')  
    soft_train_video, soft_train_label = os.path.join(data_root, 'video-unlabeled-with-test2noise'),   os.path.join(data_root, 'test1_features:chinese-hubert-large-UTT_dataset:MER2024_model:attention+utt+None_f1:0.6307_acc:0.4606_1719550924.6785953_best(128,0.2,0.0001,nodebug)_threshold0.9.csv') # 修改一下文件名即可
    test_video,  test_label  = os.path.join(data_root, 'video-unlabeled-with-test2noise'), None


    ## step1: 将semi分成两部分，一部分作为test1, 一部分作为test2
    train_names, train_emos = read_train_label(train_video, train_label)
    soft_train_names ,soft_train_emos = read_train_20000_label(soft_train_video, soft_train_label)
    test_names,  test_emos  = read_test_label(test_video, test_label)

    # 在这里 merge() 写一个merge的方法
    # merge(train_names, train_emos, soft_train_name, soft_train_emos) to (train_names, train_emos)
    train_names = train_names + soft_train_names
    train_emos = train_emos + soft_train_emos

    ## output path
    save_video = os.path.join(save_root, 'video')
    # save_label = os.path.join(save_root, 'label-6way.npz') # 改这个
    save_label = os.path.join(save_root, '1-softlabel-0.9-best(128,0.2,0.0001)-6way.npz')  # 同参数第n次-softlabel-阈值-最好参数下-6way
    if not os.path.exists(save_root):  os.makedirs(save_root)
    if not os.path.exists(save_video): os.makedirs(save_video)

    ## generate label path
    whole_corpus = {}
    for (subset, video_root, names, labels) in [ ('train', train_video, train_names, train_emos), # 这里的这个video_root没用到，不需要改
                                                 ('test1', test_video,  test_names,  test_emos)]:
        
        whole_corpus[subset] = {}
        print (f'{subset}: sample number: {len(names)}')
        for (name, label) in zip(names, labels):
            whole_corpus[subset][name] = {'emo': label}
            # copy video
            # video_path = glob.glob(os.path.join(video_root, f'{name}.*'))[0]
            # assert os.path.exists(video_path), f'video does not exist.'
            # video_name = os.path.basename(video_path)
            # new_path = os.path.join(save_video, video_name)
            #shutil.copy(video_path, new_path)

    ## 需要对视频进行加噪处理
    np.savez_compressed(save_label,
                        train_corpus=whole_corpus['train'],
                        test1_corpus=whole_corpus['test1'])


# run -d toolkit/preprocess/mer2024.py
if __name__ == '__main__':

    data_root = '/home/hao/Project/MERTools/MER2024/mer2024-dataset'
    save_root = '/data/public_datasets/MER_2024/mer2024-dataset-process'
    normalize_dataset_format(data_root, save_root)

