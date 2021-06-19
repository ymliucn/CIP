import datetime
import numpy as np
from collections import Counter


class Config:  # 选择数据集和平滑参数
    train_set_path = '../data/big_dataset/train'  # '../data/big_dataset/train'
    test_set_path = '../data/big_dataset/test'  # '../data/big_dataset/dev', '../data/big_dataset/test'
    alpha = 0.01  # 平滑参数


class Hmm:
    def __init__(self):
        self.alpha = Config.alpha  # 平滑参数
        self.word_dict = []  # 词典,最后一个词是'<UNK>'
        self.word2id = {}  # 词与词id对应的字典
        self.tag_dict = []  # 词性集合'<BOS>'开头 + 所有词性 + '<EOS>'结尾
        self.tag2id = {}  # 词性与词性id对应的字典
        self.count_tag = Counter()  # 每种词性计数
        self.tm = []  # 转移矩阵
        self.em = []  # 发射矩阵
        self.predict_data = []  # 预测集

    def estimate_transition_probability(self, train_data):  # 估计词性转移概率
        for i in range(len(train_data)):
            self.tm[0][self.tag2id[train_data[i][0][1]] - 1] += 1  # 更新('<BOS>',第一个词性)对
            self.tm[self.tag2id[train_data[i][-1][1]]][-1] += 1  # 更新(最后一个词性,'<EOS>')对
            for j in range(len(train_data[i]) - 1):
                s, t = train_data[i][j][1], train_data[i][j + 1][1]  # 前一个词性是s,后一个词性是t,即Count(s, t)
                self.tm[self.tag2id[s]][self.tag2id[t] - 1] += 1
        T = len(self.tag_dict)  # 词性数目
        for i in range(self.tm.shape[0]):
            for j in range(self.tm.shape[1]):  # q(t|s) = Count(s, t) + α / Count(s) + α × |T|
                self.tm[i][j] = (self.tm[i][j] + self.alpha) / (self.count_tag[self.tag_dict[i]] + self.alpha * T)

    def estimate_emission_probability(self, train_data):  # 使用加 α 平滑方法,估计词性生成词的概率(发射概率)
        for i in range(len(train_data)):
            for j in range(len(train_data[i])):
                t, w = train_data[i][j][1], train_data[i][j][0]  # 词性t发射出词w,即Count(t, w)
                self.em[self.tag2id[t] - 1][self.word2id[w]] += 1
        V = len(self.word_dict)  # 词数目
        for i in range(self.em.shape[0]):
            for j in range(self.em.shape[1]):  # e(w|t) = Count(w, t) + α / Count(t) + α × |V|
                self.em[i][j] = (self.em[i][j] + self.alpha) / (self.count_tag[self.tag_dict[i + 1]] + self.alpha * V)

    def fit(self, train_data):  # 在 train.conll 上使用极大似然估计方法确定模型参数
        self.word_dict = sorted(list(set([j[0] for i in train_data for j in i])))  # 词典
        self.word_dict.append('<UNK>')  # 词典加入未知词
        self.word2id = {word: index for index, word in enumerate(self.word_dict)}
        self.tag_dict = sorted(list(set([j[1] for i in train_data for j in i])))  # 词性集合
        self.tag_dict.insert(0, '<BOS>')  # 开头插入'<BOS>'
        self.tag_dict.append('<EOS>')  # 末尾插入'<EOS>'
        self.tag2id = {tag: index for index, tag in enumerate(self.tag_dict)}
        self.count_tag = Counter([j[1] for i in train_data for j in i])  # 每种词性计数,即Count(s)
        self.count_tag.update(len(train_data) * ['<BOS>', '<EOS>'])  # 每有一个句子,'<BOS>'、'<EOS>'计数加1
        self.tm = np.zeros((len(self.tag_dict) - 1, len(self.tag_dict) - 1), dtype='float')  # 转移矩阵
        self.em = np.zeros((len(self.tag_dict) - 2, len(self.word_dict)), dtype='float')  # 发射矩阵
        self.estimate_transition_probability(train_data)  # 估计词性转移概率
        self.estimate_emission_probability(train_data)  # 使用加 α 平滑方法,估计词性生成词的概率(发射概率)

    def predict(self, test_data):  # 实现 Viterbi 算法，对 dev.conll 进行词性标注
        self.tm = np.log(self.tm)  # 取对数,方便计算
        self.em = np.log(self.em)  # 取对数,方便计算

        for i in range(len(test_data)):
            sentence = ['<UNK>' if word[0] not in self.word_dict else word[0] for word in test_data[i]]
            dp = np.zeros((len(sentence), len(self.tag_dict) - 2))  # dp矩阵,去掉'<BOS>'和<EOS>'
            path = np.zeros((len(sentence), len(self.tag_dict) - 2), dtype='int')  # 保存路径
            dp[0] = self.tm[0, 0:-1] + self.em[:, self.word2id[sentence[0]]]  # 第一个词每个词性概率

            for j in range(1, len(sentence)):  # Viterbi 算法
                all_p = self.tm[1:, :-1] + self.em[:, self.word2id[sentence[j]]] + \
                        dp[j - 1].reshape(len(self.tag_dict) - 2, 1)
                dp[j] = np.max(all_p, axis=0)
                path[j] = np.argmax(all_p, axis=0)
            dp[-1] += self.tm[1:, -1]  # 加上最后一个词性转移到'<EOS>'的ln(P)
            now = np.argmax(dp[-1])  # now既是本词最有可能词性,又指向前一个词最有可能词性(反向回溯)
            predict_tags = [self.tag_dict[now + 1]]  # 先插入最可能的最后一个词性,找到路径
            for j in range(len(sentence) - 1, 0, -1):
                now = path[j][now]  # 当前词性指向的前一个最可能词性
                predict_tags.insert(0, self.tag_dict[now + 1])  # 每次前插,使反序变正序
            self.predict_data.append(predict_tags)

    def evaluate(self, test_data):
        words_with_correct_tags = 0
        for i in range(len(test_data)):
            for j in range(len(test_data[i])):
                if test_data[i][j][1] == self.predict_data[i][j]:
                    words_with_correct_tags += 1
        words_in_total = len([j[0] for i in test_data for j in i])
        accuracy = words_with_correct_tags / words_in_total
        print(f'Tagging Accuracy = {words_with_correct_tags} / {words_in_total} = ' + '{:.2%}'.format(accuracy))


def data_preprocessing(train_set_path, test_set_path):  # 把conll转为二维列表,第一维句子,第二维元组(词,词性)
    train_set = open(train_set_path, 'r', encoding='utf8')
    train_data = []
    sentence = []
    for line in train_set:
        if len(line.split()) > 1:  # 不是空行
            word, tag = line.split()[1], line.split()[3]
            sentence.append((word, tag))
        else:
            train_data.append(sentence)
            sentence = []
    train_set.close()
    test_set = open(test_set_path, 'r', encoding='utf8')
    test_data = []
    for line in test_set:
        if len(line.split()) > 1:  # 不是空行
            word, tag = line.split()[1], line.split()[3]
            sentence.append((word, tag))
        else:
            test_data.append(sentence)
            sentence = []
    test_set.close()
    return train_data, test_data


def main():  # 实现一个二元（一阶）隐马尔科夫模型，做词性标注任务
    start_time = datetime.datetime.now()  # 程序开始时间
    train_data, test_data = data_preprocessing(Config.train_set_path, Config.test_set_path)  # 数据预处理
    hmm_model = Hmm()
    hmm_model.fit(train_data)  # 在 train.conll 上使用极大似然估计方法确定模型参数
    hmm_model.predict(test_data)  # 实现 Viterbi 算法，对 dev.conll 进行词性标注
    hmm_model.evaluate(test_data)  # 在 dev.conll 上评价模型的词性准确率
    end_time = datetime.datetime.now()  # 程序结束时间
    print(f'总共耗时：{end_time - start_time}')  # 输出程序运行时间


if __name__ == '__main__':
    main()
