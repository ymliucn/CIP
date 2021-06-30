import random
from utils import *
from config import *
from collections import defaultdict


class LogLinearModel:
    def __init__(self):
        self.tag2id = {}  # 词性的字典, 每一个词性对应一个唯一的数字
        self.tag_num = 0  # 词性种类数
        self.feature2id = {}  # 部分特征空间的字典, 每一个部分特征对应一个唯一的数字
        self.feature_num = 0
        self.w = []  # 特征权重
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
        partial_feature_space = set()
        for sentence in train_set.data:
            for index in range(len(sentence)):
                feature_list = self.instantiate_feature_template(sentence, index)
                partial_feature_space.update(feature_list)
        self.feature2id = {feature: ID for ID, feature in enumerate(list(partial_feature_space))}
        self.feature_num = len(partial_feature_space)

        with open(result_path, 'a', encoding='utf-8') as result:
            result.write("\n二、创建特征空间\n")  # 开始训练
            result.write(f'特征空间维度=词性种类数×部分特征数={self.tag_num}×{self.feature_num}'
                         f'={self.tag_num * self.feature_num}\n')

    def evaluate(self, data_set):
        correct = 0
        for sentence in data_set.data:
            for index in range(len(sentence)):
                feature_list = self.instantiate_feature_template(sentence, index)
                score = np.zeros(self.tag_num)
                for feature in feature_list:
                    feature_id = self.feature2id.get(feature)
                    if feature_id:
                        for tag in range(self.tag_num):
                            score[tag] += self.w[feature_id][tag]
                predict_tag = np.argmax(score)
                true_tag = self.tag2id[sentence[index][1]]
                if predict_tag == true_tag:
                    correct += 1
        total = data_set.total_word_num
        accuracy = correct / total
        return correct, total, accuracy

    def MBGD(self, train_set, dev_set, test_set):
        with open(result_path, 'a', encoding='utf-8') as result:
            result.write("\n三、LogLinearModel开始训练\n")  # 开始训练
        dev_max_accuracy = 0  # 验证集最大正确率
        best_iteration = 0
        self.w = np.zeros((self.feature_num, self.tag_num))
        g = defaultdict(float)
        k = 0
        B = self.batch_size
        b = 0
        n = train_set.total_word_num / self.batch_size
        step = 0
        global_step = 100000
        for iteration in range(1, max_iterations + 1):
            if shuffle:  # 是否打乱训练集
                random.shuffle(train_set.data)
            for sentence in train_set.data:
                for index in range(len(sentence)):
                    feature_list = self.instantiate_feature_template(sentence, index)
                    feature_id_list = []
                    score = np.zeros(self.tag_num)
                    for feature in feature_list:
                        feature_id = self.feature2id[feature]
                        feature_id_list.append(feature_id)
                        for tag in range(self.tag_num):
                            score[tag] += self.w[feature_id][tag]
                    Y = softmax(score)
                    true_tag = self.tag2id[sentence[index][1]]
                    for feature_id in feature_id_list:
                        g[feature_id] -= Y
                        g[feature_id][true_tag] += 1
                    if B == b:
                        eta = self.init_eta * self.decay_rate ** (step / global_step)  # 步长衰减
                        self.w *= (1 - eta * self.C / n)  # 加入正则项, 权重衰减
                        for feature_id, gradient in g.items():
                            self.w[feature_id] += eta * g[feature_id]
                        k += 1
                        b = 0
                        g = defaultdict(float)
                        step += 1
                    b += 1
            with open(result_path, 'a', encoding='utf-8') as result:
                result.write(f'\niteration {iteration} 训练完成\n')
            train_correct, train_total, train_accuracy = self.evaluate(train_set)
            dev_correct, dev_total, dev_accuracy = self.evaluate(dev_set)
            with open(result_path, 'a', encoding='utf-8') as result:
                result.write(f'训练集{train_set.file_name}：'
                             f'Accuracy = {train_correct} / {train_total} = {train_accuracy}\n')
                result.write(f'验证集{dev_set.file_name}：'
                             f'Accuracy = {dev_correct} / {dev_total} = {dev_accuracy}\n')
            if test_set:
                test_correct, test_total, test_accuracy = self.evaluate(test_set)
                with open(result_path, 'a', encoding='utf-8') as result:
                    result.write(f'测试集{test_set.file_name}：'
                                 f'Accuracy = {test_correct} / {test_total} = {test_accuracy}\n')
            if dev_accuracy > dev_max_accuracy:
                dev_max_accuracy = dev_accuracy
                best_iteration = iteration
            if iteration - best_iteration >= max_no_rise:
                with open(result_path, 'a', encoding='utf-8') as result:
                    result.write(f'\n验证集正确率连续{max_no_rise}轮没有上升,LogLinearModel结束训练。')
                break
        with open(result_path, 'a', encoding='utf-8') as result:
            result.write(f'\n验证集最大正确率：iteration：{best_iteration},  Max Accuracy：{dev_max_accuracy}\n')

    def fit(self, train_set, dev_set, test_set):
        self.tag2id = train_set.tag2id
        self.tag_num = len(self.tag2id)
        self.create_feature_space(train_set)  # 确定feature space, 即收集训练数据中所有的特征
        self.MBGD(train_set, dev_set, test_set)  # 通过mini-batch梯度下降更新权重
