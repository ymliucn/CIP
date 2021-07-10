import random
import numpy as np
from config import *
from scipy.special import logsumexp
from collections import defaultdict


class CRF:
    def __init__(self):
        self.tag2id = {}  # 词性的字典, 每一个词性对应一个唯一的数字
        self.tag_num = 0  # 词性种类数
        self.feature2id = {}  # 部分特征空间的字典, 每一个部分特征对应一个唯一的数字
        self.feature_num = 0  # 特征数
        self.w = []  # 特征权重
        self.trans_w = []  # 转移权重矩阵
        self.init_eta = init_eta  # 初始更新步长
        self.decay_rate = decay_rate  # 步长衰减速率
        self.C = C  # 正则化权重
        self.batch_size = batch_size  # 批量大小

    @staticmethod
    def instantiate_feature_template(sentence, index):
        """
        实例化特征模板, 对于每一个实例, 根据特征模板列表, 得到具体的特征列表
        :param sentence: 句子
        :param index: 实例在句子中的位置
        :return: feature_list, 该实例对应的部分特征列表
        """
        current_word = sentence[index][0]  # 当前词
        len_current_word = len(current_word)  # 当前词的长度
        previous_word = sentence[index - 1][0] if index != 0 else '<BOS>'  # 当前词的前一个词
        next_word = sentence[index + 1][0] if index != len(sentence) - 1 else '<EOS>'  # 当前词的后一个词
        previous_word_last_c = previous_word[-1] if index != 0 else '<BOS>'  # 当前词的前一个词的最后一个字
        next_word_first_c = next_word[0] if index != len(sentence) - 1 else '<EOS>'  # 当前词的后一个词的第一个字
        current_word_first_c = current_word[0]  # 当前词的第一个字
        current_word_last_c = current_word[-1]  # 当前词的最后一个字
        current_word_middle_c = current_word[1:-1]  # 当前词中间的字
        feature_list = ['02:' + current_word,
                        '03:' + previous_word,
                        '04:' + next_word,
                        '05:' + current_word + '◦' + previous_word_last_c,
                        '06:' + current_word + '◦' + next_word_first_c,
                        '07:' + current_word_first_c,
                        '08:' + current_word_last_c
                        ]
        for c in current_word_middle_c:
            feature_list.append('09:' + c)
            feature_list.append('10:' + current_word_first_c + '◦' + c)
            feature_list.append('11:' + current_word_last_c + '◦' + c)
        if len_current_word == 1:
            feature_list.append('12:' + current_word + '◦' + previous_word_last_c + '◦' + next_word_first_c)
        for k, c in enumerate(current_word[:-1]):
            next_c = current_word[k + 1]
            if c == next_c:
                feature_list.append('13:' + c + '◦' + 'consecutive')
        max_k = min(4, len_current_word)
        for k in range(max_k):
            feature_list.append('14:' + current_word[:k + 1])
            feature_list.append('15:' + current_word[len_current_word - k - 1:])
        return feature_list

    def create_feature_space(self, train_set):
        """
        构造所有训练数据 train_set 中出现的所有特征构成的集合, 即特征空间
        :param train_set: 训练集数据
        :return:
        """
        print("\n二、创建特征空间")  # 开始训练
        partial_feature_space = set()
        for sentence in train_set.data:
            for index in range(len(sentence)):
                feature_list = self.instantiate_feature_template(sentence, index)
                partial_feature_space.update(feature_list)
        self.feature2id = {feature: ID for ID, feature in enumerate(list(partial_feature_space))}
        self.feature_num = len(partial_feature_space)
        print(f'特征空间维度 = 部分特征数 × 词性种类数 = '
              f'{self.feature_num} × {self.tag_num} + {self.tag_num + 1} × {self.tag_num} '
              f'= {self.tag_num * self.feature_num + (self.tag_num + 1) * self.tag_num}')

    def Viterbi(self, dp, word_num):  # 利用维特比算法估计最可能的词性序列
        path = np.zeros((word_num, self.tag_num), dtype='int')  # 保存路径
        dp[0] += self.trans_w[0]  # 加上开始词性转移到第一个词性的得分
        for word in range(1, word_num):
            all_score = self.trans_w[1:, :] + dp[word - 1].reshape(-1, 1)
            dp[word] += np.max(all_score, axis=0)
            path[word] = np.argmax(all_score, axis=0)
        predict_tag = np.argmax(dp[-1])  # predict_tag是本词最有可能词性, 在path矩阵中它作为索引指向前一个最有可能词性
        predict_Y = [predict_tag]  # 反向回溯, 先插入最可能的最后一个词性, 找到路径
        for word in range(word_num - 1, 0, -1):
            predict_tag = path[word][predict_tag]  # 本词最有可能词性
            predict_Y.insert(0, predict_tag)  # 每次前插, 反向回溯
        return predict_Y

    def forward(self, Score, word_num):  # 前向算法
        """
        :param Score: 每个词中每个词性部分特征的得分和(不包括 01:转移特征 )
        :param word_num: 句子的词数
        :return: forward_score是一个矩阵,第一维是每个词,第二维是传播到该词时,每一条路径的得分和(log sum exp)
        """
        forward_score = np.zeros((word_num, self.tag_num))
        forward_score[0] = self.trans_w[0] + Score[0]  # 第一个词的前一个词性全是开始词性
        for i in range(1, word_num):
            all_score = forward_score[i - 1] + (self.trans_w[1:, :] + Score[i]).T
            forward_score[i] = logsumexp(all_score, axis=1)  # 维特比算法求每列最大值,这里求和
        return forward_score

    def backward(self, Score, word_num):  # 后向算法
        """
        :param Score: 每个词中每个词性部分特征的得分和(不包括 01:转移特征 )
        :param word_num: 句子的词数
        :return: backward_score是一个矩阵,第一维是每个词,第二维是该词传到最后一个词时,每一条路径的得分和(log sum exp)
        """
        backward_score = np.zeros((word_num, self.tag_num))
        for i in range(word_num - 2, -1, -1):
            all_score = self.trans_w[1:, :] + Score[i + 1] + backward_score[i + 1]
            backward_score[i] = logsumexp(all_score, axis=1)  # 维特比算法求每列最大值,这里求和
        return backward_score

    def evaluate(self, data_set):
        correct = 0
        for sentence in data_set.data:
            word_num = len(sentence)
            true_Y = []  # 真实的词性序列
            dp = []
            for index in range(word_num):
                feature_list = self.instantiate_feature_template(sentence, index)
                feature_id_list = []
                score = np.zeros(self.tag_num)
                for tag in range(self.tag_num):
                    for feature in feature_list:
                        feature_id = self.feature2id.get(feature)
                        if feature_id:
                            feature_id_list.append(feature_id)
                            score[tag] += self.w[feature_id][tag]
                dp.append(score)
                true_Y.append(self.tag2id[sentence[index][1]])
            predict_Y = self.Viterbi(dp, word_num)
            correct += sum([true_Y[i] == predict_Y[i] for i in range(word_num)])
        total = data_set.total_word_num
        accuracy = correct / total
        return correct, total, accuracy

    def MBGD(self, train_set, dev_set, test_set):
        print("\n三、CRF开始训练")  # 开始训练
        dev_max_accuracy = 0  # 验证集最大正确率
        best_iteration = 0  # 验证集最大正确率所在的轮数
        train_correct = 0  # 训练集正确词数
        train_total = train_set.total_word_num  # 训练集总词数
        self.w = np.zeros((self.feature_num, self.tag_num))  # 权重矩阵
        self.trans_w = np.zeros((self.tag_num + 1, self.tag_num))
        g = defaultdict(float)
        g_trans_w = np.zeros((self.tag_num + 1, self.tag_num))
        B = self.batch_size
        b = 0
        n = train_set.total_word_num / self.batch_size
        step = 0
        global_step = 100000
        for iteration in range(1, max_iterations + 1):
            if shuffle:  # 是否打乱训练集
                random.shuffle(train_set.data)
            for sentence in train_set.data:
                word_num = len(sentence)
                true_Y = []  # 真实的词性序列
                Score = []
                all_features = []
                for index in range(word_num):
                    feature_list = self.instantiate_feature_template(sentence, index)
                    feature_id_list = []
                    score = np.zeros(self.tag_num)
                    for feature in feature_list:
                        feature_id = self.feature2id[feature]
                        feature_id_list.append(feature_id)
                        for tag in range(self.tag_num):
                            score[tag] += self.w[feature_id][tag]
                    all_features.append(feature_id_list)
                    Score.append(score)
                    true_Y.append(self.tag2id[sentence[index][1]])
                predict_Y = self.Viterbi(Score, word_num)  # 根据维特比算法预测的词性序列
                train_correct += sum([true_Y[i] == predict_Y[i] for i in range(word_num)])
                forward_score = self.forward(Score, word_num)
                backward_score = self.backward(Score, word_num)
                Z_S = logsumexp(forward_score[-1])
                p = np.exp(self.trans_w[0] + Score[0] + backward_score[0] - Z_S)
                feature_id_list = all_features[0]
                true_tag = true_Y[0]
                for feature_id in feature_id_list:
                    g[feature_id] -= p
                    g[feature_id][true_tag] += 1
                g_trans_w[0] -= p
                g_trans_w[0][true_tag] += 1
                for i in range(1, word_num):
                    true_tag, last_true_tag = true_Y[i], true_Y[i - 1]
                    feature_id_list = all_features[i]
                    all_score = self.trans_w[1:, :] + Score[i]
                    all_p = np.exp(forward_score[i - 1].reshape(-1, 1) + all_score + backward_score[i] - Z_S)
                    sum_p = sum(all_p)
                    for feature_id in feature_id_list:
                        g[feature_id] -= sum_p
                        g[feature_id][true_tag] += 1
                    g_trans_w[1:] -= all_p
                    g_trans_w[last_true_tag + 1][true_tag] += 1
                    if B == b:
                        eta = self.init_eta * self.decay_rate ** (step / global_step)  # 步长衰减
                        self.w *= (1 - eta * self.C / n)  # 加入正则项, 权重衰减
                        self.trans_w *= (1 - eta * self.C / n)
                        for feature_id, gradient in g.items():
                            self.w[feature_id] += eta * g[feature_id]  # 更新特征权重矩阵
                        self.trans_w += eta * g_trans_w
                        step += 1
                        b = 0
                        g = defaultdict(float)
                        g_trans_w = np.zeros((self.tag_num + 1, self.tag_num))
                    b += 1
            print(f'\niteration {iteration} 训练完成')
            train_accuracy = train_correct / train_total
            print(f'训练集{train_set.file_name}：Accuracy = {train_correct} / {train_total} = {train_accuracy}')
            train_correct = 0
            dev_correct, dev_total, dev_accuracy = self.evaluate(dev_set)
            print(f'验证集{dev_set.file_name}：Accuracy = {dev_correct} / {dev_total} = {dev_accuracy}')
            if test_set:
                test_correct, test_total, test_accuracy = self.evaluate(test_set)
                print(f'测试集{test_set.file_name}：Accuracy = {test_correct} / {test_total} = {test_accuracy}')
            if dev_accuracy > dev_max_accuracy:
                dev_max_accuracy = dev_accuracy
                best_iteration = iteration
            if iteration - best_iteration >= max_no_rise:
                print(f'\n验证集正确率连续{max_no_rise}轮没有上升,CRF结束训练。')
                break
        print(f'\n验证集最大正确率：iteration：{best_iteration},  Max Accuracy：{dev_max_accuracy}')

    def fit(self, train_set, dev_set, test_set):
        self.tag2id = train_set.tag2id
        self.tag_num = len(self.tag2id)
        self.create_feature_space(train_set)  # 确定feature space, 即收集训练数据中所有的特征
        self.MBGD(train_set, dev_set, test_set)
