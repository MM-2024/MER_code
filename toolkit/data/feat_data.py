import torch
import numpy as np
from torch.utils.data import Dataset
from toolkit.utils.read_data import *
import random

class Data_Feat(Dataset):
    def __init__(self, args, names, labels): # label由两个标签构成，'emo'=情感标签，'val'=情感强度标签, 后者恒等于 -10， 就是说没啥用

        # analyze path
        self.names = names
        self.labels = labels
        feat_root  = my_config.PATH_TO_FEATURES[args.dataset]
        if args.snr is None: # 通过snr控制特征读取位置
            audio_root = os.path.join(feat_root, args.audio_feature)
            text_root  = os.path.join(feat_root, args.text_feature )
            video_root = os.path.join(feat_root, args.video_feature)
        else:
            # data2vec-audio-base-960h-UTT -> data2vec-audio-base-960h-noisesnrmix-UTT
            # eGeMAPS_UTT -> eGeMAPS_noisesnrmix_UTT
            audio_root = os.path.join(feat_root, args.audio_feature[-4].join([args.audio_feature[:-4], args.snr, 'UTT']))
            text_root  = os.path.join(feat_root, args.text_feature[-4].join([args.text_feature[:-4], args.snr, 'UTT']))
            video_root = os.path.join(feat_root, args.video_feature[-4].join([args.video_feature[:-4], args.snr, 'UTT']))
        print (f'audio feature root: {audio_root}')
        print (f'video feature root: {video_root}')
        print (f'text feature root: {text_root}')
        # --------------- temporal test ---------------
        # for name in names: assert os.path.exists(os.path.join(audio_root, name+'.npy'))

        # analyze params
        self.feat_type = args.feat_type
        self.feat_scale = args.feat_scale # 特征预压缩
        assert self.feat_scale >= 1
        assert self.feat_type in ['utt', 'frm_align', 'frm_unalign']

        print(type(names))
        #if type(names) == list:
        print('here',len(names)) # here 5030
        # 检查特征文件是否存在，如果不存在则从列表中移除
        existing_names = []
        for name in self.names:
            audio_path = os.path.join(audio_root, name + '.npy')
            text_path = os.path.join(text_root, name + '.npy')
            video_path = os.path.join(video_root, name + '.npy')
            # 打印路径以帮助诊断问题
            #print(f"Checking: {audio_path}, {text_path}, {video_path}")
            if os.path.exists(audio_path)  and os.path.exists(video_path) and os.path.exists(text_path):
                existing_names.append(name)
            else:
                print(f"Feature not found for {name}, skipping.")
                pass

        # 更新names列表为只包含存在的特征文件的名称
        self.names = existing_names
        # 现在使用更新后的names列表调用func_read_multiprocess
        audios, self.adim = func_read_multiprocess(audio_root, self.names, read_type='feat')
        texts,  self.tdim = func_read_multiprocess(text_root,  self.names, read_type='feat')
        videos, self.vdim = func_read_multiprocess(video_root, self.names, read_type='feat')

        ## read batch (reduce collater durations)
        # step1: pre-compress features
        audios, texts, videos = feature_scale_compress(audios, texts, videos, self.feat_scale)
        # step2: align to batch
        if self.feat_type == 'utt': # -> 每个样本每个模态的特征压缩到句子级别
            audios, texts, videos = align_to_utt(audios, texts, videos)
        elif self.feat_type == 'frm_align':
            audios, texts, videos = align_to_text(audios, texts, videos) # 模态级别对齐
            audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        elif self.feat_type == 'frm_unalign':
            audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        self.audios, self.texts, self.videos = audios, texts, videos

 
    def __len__(self):
        return len(self.names)



    def __getitem__(self, index):
        
        try:
            instance = dict(
                audio = self.audios[index],
                text  = self.texts[index],
                video = self.videos[index],
                emo   = self.labels[index]['emo'],
                val   = self.labels[index]['val'],
                name  = self.names[index],
            )
            if index == 19999:
                print(instance)
            return instance
        except Exception as e:
            print(f"Error with index {index}: {e}. Selecting a random sample instead.")
            new_index = random.randint(0, len(self.names) - 1)
            return self.__getitem__(new_index)
    

    def collater(self, instances):
        audios = [instance['audio'] for instance in instances]
        texts  = [instance['text']  for instance in instances]
        videos = [instance['video'] for instance in instances]

        batch = dict(
            audios = torch.FloatTensor(np.array(audios)),
            texts  = torch.FloatTensor(np.array(texts)),
            videos = torch.FloatTensor(np.array(videos)),
        )
        
        emos  = torch.LongTensor([instance['emo']  for instance in instances])
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]

        return batch, emos, vals, names
    

    def get_featdim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim
    