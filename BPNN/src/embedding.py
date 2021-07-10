"""
读入数据预训练的词向量文件,增加上下文窗口
"""
import numpy as np
from config import *


class Embedding:
    def __init__(self, file_path):
        self.unk = '<UNK>'  # 不在词典中的词
        self.bos = '<BOS>'  # 句首token
        self.eos = '<EOS>'  # 句尾token
        self.unk_id = 0
        self.bos_id = 0
        self.eos_id = 0
        self.marks = [self.unk, self.bos, self.eos]  # 未登录词、句首、句尾标记
        self.window = window  # 上下文窗口
        self.embedding_dim = embedding_dim  # embedding词向量的维度
        self.word_num = 0  # embedding矩阵中所有的单词

        self.word2id, self.id2word, self.embedding_matrix = self.word_embedding(file_path)

        with open(result_path, 'a', encoding='utf-8') as result:
            result.write(f'初始词向量矩阵：词数：{self.word_num},  词向量维度：{self.embedding_dim}\n')

    def word_embedding(self, file_path):
        """
        读取词向量文件,转为词向量矩阵保存
        :param file_path: 预训练好的词向量文件路径
        :return:
        word2id:词->id字典
        id2word:id->词字典
        embedding_matrix:词向量矩阵
        """
        word2id = {}
        id2word = {}
        embedding_matrix = []
        f = open(file_path, 'r', encoding='utf-8')  # giga.100.txt预训练的词向量文件
        index = 0
        for index, line in enumerate(f):
            word_vector = line.strip().split()  # 每个列表第一个元素是词,后面是对应的词向量
            embedding_matrix.append(list(map(float, word_vector[1:])))  # 词向量加入嵌入矩阵
            word2id[word_vector[0]] = index  # 保存word->id字典映射
            id2word[index] = word_vector[0]  # 保存id->word字典映射
            index += 1
        f.close()
        for mask in self.marks:  # 把'<UNK>', '<BOS>', '<EOS>'加入embedding矩阵,其词向量随机初始化
            if mask not in word2id.keys():  # 若'<UNK>', '<BOS>', '<EOS>'不在word->id字典中,就添加
                word2id[mask] = index
                id2word[index] = mask
                np.random.randn(self.embedding_dim) / np.sqrt(self.embedding_dim)
                embedding_matrix.append((np.random.randn(self.embedding_dim) / np.sqrt(self.embedding_dim)).tolist())
                index += 1
        embedding_matrix = np.array(embedding_matrix)
        self.word_num = len(word2id)
        self.eos_id = word2id[self.eos]
        self.bos_id = word2id[self.bos]
        self.unk_id = word2id[self.unk]
        return word2id, id2word, embedding_matrix

    def extend_embedding(self, data):
        """
        利用新加入的数据扩展embedding_matrix,就是数据集中未出现在嵌入矩阵的词,
        用embedding_dim维向量随机初始化,并加入embedding_matrix中
        :param data: 新加入的数据
        :return:只更新self变量,无返回值
        """
        unk_words = [word for word in data.word_dict if word not in self.word2id.keys()]
        add_num = len(unk_words)
        index = self.word_num
        for unk_word in unk_words:
            self.word2id[unk_word] = index
            self.id2word[index] = unk_word
            self.embedding_matrix = np.row_stack(
                (self.embedding_matrix, np.random.randn(self.embedding_dim) / np.sqrt(self.embedding_dim)))
            index += 1
        self.word_num = len(self.word2id)
        with open(result_path, 'a', encoding='utf-8') as result:
            result.write(f'通过数据集{data.file_name}扩展后,  词向量矩阵新增词数：{add_num},  扩展后词数：{self.word_num}\n')

    def add_context_window(self, data, unique_tag_num, tag2id):
        """
        把数据集转换为 total_word_num 维的样本,每个样本的标签是词性, 而特征是window个词(当前词以及前后window//2个词)
        在每个句子前加 half//2 个句子开始标记,每个句子后加 half//2 个句子结束标记
        其目的一是方便取上下文词,两个列表都可以从0开始遍历
        二是开始及结束的half//2个词(包括词数不足half//2时),对应词性依然有window个词作为特征
        :param data: 数据集
        :param unique_tag_num: 词性数
        :param tag2id: 词性与id转换
        :return: 处理好的样本列表
        """
        dataset = []
        half_window = self.window // 2  # window表示上下文窗口大小
        for sentence in data.data:
            word_id_list = [self.bos_id] * half_window + \
                           [self.word2id.get(word[0], self.unk_id) for word in sentence] + \
                           [self.eos_id] * half_window
            tag_id_list = [tag2id.get(word[1]) for word in sentence]
            for i in range(len(tag_id_list)):
                x = word_id_list[i:i + self.window]
                y = self.one_hot(tag_id_list[i], unique_tag_num)
                dataset.append((x, y))
        with open(result_path, 'a', encoding='utf-8') as result:
            result.write(f'数据集{data.file_name}增加上下文窗口：数据集样本维度：{len(dataset)}'
                         f',  每个样本中词性对应的上下文词数目：{self.window}\n')
        return dataset

    @staticmethod
    def one_hot(index, unique_tag_num):
        """
        对每个标签(也就是词性)进行one-hot编码,分类标签无法成为神经网络输出,需要转为向量形式(如 1 -> [0...0...1]的转置)
        :param index:词性在词性列表中的索引，位置
        :param unique_tag_num:每个one-hot矩阵的维度不重复的词性数目
        :return:一个 unique_tag_num × 1 的one-hot矩阵
        """
        tag_vector = np.zeros((unique_tag_num, 1))
        tag_vector[index][0] = 1.0
        return tag_vector
