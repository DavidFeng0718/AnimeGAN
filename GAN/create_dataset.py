import os
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid
import torch.utils.data.dataset as Dataset


class My_dataset(Dataset.Dataset):
    """
    继承自torch.utils.data.Dataset
    """

    def __init__(self, path, transform):
        """
            初始化数据集
            参数：
                path: 图片数据集所在目录
                transform: 图像预处理函数
            成员变量：
                self.path
                    保存数据集路径
                self.loc_list
                    保存目录下所有图片文件名
                self.transform
                    保存预处理函数，后续在 __getitem__()
                    中对每张图片进行处理

        """
        self.path = path
        loc_list = os.listdir(self.path)
        self.loc_list = loc_list
        self.tranform = transform

    def __getitem__(self, index):
        """
        根据索引读取一张图片，转化为PIL格式，然后再转回Tensor格式，最后返回给DataLoader
        """
        loc_data = os.path.join(self.path, self.loc_list[index])
        data = Image.open(loc_data)
        data = np.array(data)
        data = Image.fromarray(data)  # 若导入图像不为PIL格式则需要转换
        data = self.tranform(data)
        return data

    """
    数据集长度
    """

    def __len__(self):

        return len(self.loc_list)


"""
    (-0.75,0.35,-0.1) 
          ↓
  (0.125,0.675,0.45)      
          ↓
(31.875,172.125,114.75)
          ↓
 (172.125,114.75,31.875)

"""

def save_img(tensor, fp):
    '''

    tensor：一个tensor数据结构，里面存着需要保存的tensor数据
    fp：保存路径

    '''

    grid = make_grid(tensor) #多张图片拼大图 (tensor 里包含很多图（如64张)
    ndarr = (grid.mul(0.5).add_(0.5)).mul(255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()  #先把数据集从-1，1变成0，1，然后再把他乘255变成RGB范围0，255，然后再改维度顺（C，H，W），

    im = Image.fromarray(ndarr, mode='RGB') #生成图
    im.save(fp) #保存图

