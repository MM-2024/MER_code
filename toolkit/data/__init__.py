from torch.utils.data import Dataset

from .feat_data import Data_Feat
from .feat_data_topn import Data_Feat_TOPN

import random
# 目标：输入 (names, labels, data_type)，得到所有特征与标签
class get_datasets(Dataset):

    def __init__(self, args, names, labels):

        MODEL_DATASET_MAP = {
            
            # 解析特征
            'attention': Data_Feat,
            'lf_dnn': Data_Feat,
            'lmf': Data_Feat,
            'misa': Data_Feat,
            'mmim': Data_Feat,
            'tfn': Data_Feat,
            'mfn': Data_Feat,
            'graph_mfn': Data_Feat,
            'ef_lstm': Data_Feat, 
            'mfm': Data_Feat,
            'mctn': Data_Feat,
            'mult': Data_Feat,


            # 兼容多特征输入
            'attention_topn': Data_Feat_TOPN,
        }

        self.dataset_class = MODEL_DATASET_MAP[args.model] # Data_Feat
        self.dataset = self.dataset_class(args, names, labels)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        dataset_length = len(self.dataset)
        tried_indices = set()  # 用于存储已尝试的索引，避免重复尝试

        while len(tried_indices) < dataset_length:
            try:
                # 尝试获取当前索引的项
                chosen_index = (index + random.randint(0, dataset_length - 1)) % dataset_length
                if chosen_index in tried_indices:
                    continue  # 如果这个索引已经尝试过，跳过
                tried_indices.add(chosen_index)
                return self.dataset.__getitem__(chosen_index)
            except Exception as e:
                # 如果出现异常，打印异常信息（可选）并尝试随机索引
                print(f"Error at index {chosen_index}: {e}")
                continue
        # 如果所有索引都尝试失败，抛出异常
        raise ValueError("Unable to retrieve any item from the dataset.")

    def collater(self, instances):
        return self.dataset.collater(instances)
         
    def get_featdim(self):
        return self.dataset.get_featdim()