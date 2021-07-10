"""
构建小批量样本
"""
import random
from config import *


class DataLoader:
    def __init__(self, data):
        self.shuffle = shuffle
        self.batch_size = batch_size
        if self.shuffle:  # 是否打乱数据集
            random.shuffle(data)
        self.batch_list = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
