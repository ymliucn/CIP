"""
读入数据文件
"""
import os
from config import *


class Dataset:
    def __init__(self, file_path):
        self.file_name = os.path.basename(file_path)
        self.data = []  # 所有的数据,是二维列表,第一维句子,第二维(word,tag)对
        self.tag_dict = []  # 有序词性集合
        self.tag2id = {}  # 词性与词性id对应的字典
        self.sentence_num = 0  # 句子数
        self.total_word_num = 0  # 总词数
        self.unique_tag_num = 0  # 词性种类数

        self.load(file_path)  # 加载数据

        with open(result_path, 'a', encoding='utf-8') as result:
            result.write(f'数据集{self.file_name}：句子数：{self.sentence_num},  总词数：{self.total_word_num}'
                         f',  词性种类数：{self.unique_tag_num}\n')

    def load(self, file_path):
        """
        把conll中的有用数据按照需要的结构来保存
        :param file_path: 数据集的路径
        :return:0
        """
        corpus = open(file_path, 'r', encoding='utf-8')
        sentence = []
        tag_set = set()
        for line in corpus:
            if len(line) >= 5:
                word, tag = line.split()[1], line.split()[3]
                tag_set.add(tag)
                sentence.append((word, tag))
                self.total_word_num += 1
            else:
                self.data.append(sentence)
                sentence = []
        corpus.close()
        self.tag_dict = sorted(list(tag_set))
        self.tag2id = {tag: index for index, tag in enumerate(self.tag_dict)}
        self.sentence_num = len(self.data)
        self.unique_tag_num = len(tag_set)
