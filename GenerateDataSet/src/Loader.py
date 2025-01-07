import os
from glob import glob       # 可以用于查找符合特定规则和文件路径名
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ct_dataset(Dataset):              # 一个ct_dataset的类 用于加载数据
    def __init__(self, mode, saved_path, test_patient):
        assert mode in ['train', 'test', 'all'], "mode is 'train', 'test' or 'all'"

        # 获取所有图像的完整路径名  按结尾不同区分输入和输出
        # 并将所有路径按照顺序排列
        input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))              # os.path.join()函数用于路径拼接文件路径，可以传入多个路径
        target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))

        if mode == 'train':
            # 剔除不符合条件的数据
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]

        if mode == 'test':
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]

        if mode == 'all':
            input_ = [f for f in input_path]
            target_ = [f for f in target_path]

        self.input_ = input_
        self.target_ = target_

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        input_img, target_img = np.load(input_img), np.load(target_img)

        return (input_img, target_img)

def Get_Loader(mode='train', saved_path=None, test_patient='L506', batch_size=32, shuffled=True, num_workers=6):
    # 构造CT dataset对象
    dataset_ = ct_dataset(mode, saved_path, test_patient)
    # 按照dataset对象内容读取数据   这里要熟悉data_loader的使用方法
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=shuffled, num_workers=num_workers)
    return data_loader