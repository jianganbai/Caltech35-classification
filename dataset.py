import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch


class tiny_caltech35(Dataset):
    def __init__(self, transform=None, used_data=['train'], wrong_prop=0):
        self.train_dir = 'dataset/train/'
        self.addition_dir = 'dataset/addition/'
        self.val_dir = 'dataset/val/'
        self.test_dir = 'dataset/test/'
        self.used_data = used_data
        for x in used_data:
            assert x in ['train', 'addition', 'val', 'test']
        self.transform = transform

        self.samples, self.annotions = self._load_samples()  # 图片路径+标签
        self.class_num = max(self.annotions)+1
        if wrong_prop > 0:
            assert wrong_prop < 1  # 错误标签的比例应小于1
            wrong_num = int(wrong_prop*len(self.annotions))
            self.wrong_loc = np.random.choice(len(self.annotions), size=wrong_num, replace=False)
            for loc in self.wrong_loc:
                wrong_label = np.random.randint(self.class_num, size=1)
                while(wrong_label == self.annotions[loc]):
                    wrong_label = np.random.randint(self.class_num, size=1)
                self.annotions[loc] = wrong_label.tolist()[0]

    def _load_samples_one_dir(self, dir='dataset/train/'):
        samples, annotions = [], []
        if 'test' not in dir:  # 针对训练集和验证集
            sub_dir = os.listdir(dir)
            for i in sub_dir:  # 遍历不同类别，i是类别
                tmp = os.listdir(os.path.join(dir, i))
                samples += [os.path.join(dir, i, x) for x in tmp]  # 记录图像路径
                annotions += [int(i)] * len(tmp)  # 记录标签（0-34）
        else:
            with open(os.path.join(self.test_dir, 'annotions.txt'), 'r') as f:
                tmp = f.readlines()  # annotions内部为：图片名，标签
            for i in tmp:
                path, label = i.split(',')[0], i.split(',')[1]
                samples.append(os.path.join(self.test_dir, path))
                annotions.append(int(label))
        return samples, annotions

    def _load_samples(self):  # 载入图片路径+标签
        samples, annotions = [], []
        for i in self.used_data:
            if i == 'train':
                tmp_s, tmp_a = self._load_samples_one_dir(dir=self.train_dir)
            elif i == 'addition':
                tmp_s, tmp_a = self._load_samples_one_dir(dir=self.addition_dir)
            elif i == 'val':
                tmp_s, tmp_a = self._load_samples_one_dir(dir=self.val_dir)
            elif i == 'test':
                tmp_s, tmp_a = self._load_samples_one_dir(dir=self.test_dir)
            else:
                print('error used_data!!')
                exit(0)
            samples += tmp_s
            annotions += tmp_a
        return samples, annotions

    def __getitem__(self, index):
        img_path, img_label = self.samples[index], self.annotions[index]
        img = self._loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_label

    def _loader(self, img_path):  # 读入图片
        return Image.open(img_path).convert('RGB')

    def __len__(self):
        return len(self.samples)

    def get_wrong_loc(self):
        return self.wrong_loc

    def get_annotions(self):
        return self.annotions
