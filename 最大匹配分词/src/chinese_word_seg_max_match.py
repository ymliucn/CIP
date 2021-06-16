def create_dict():  # 给一个人工分好词的文件data.conll,构建一个词典,输出到word.dict中
    data_conll = open('../data/data.conll', 'r', encoding='utf-8')
    word_dict = open('../data/word.dict', 'w', encoding='utf-8')
    word_set = set()
    for line in data_conll:
        if len(line) > 1:
            if line.split()[1] not in word_set:
                word_set.add(line.split()[1])
                print(line.split()[1], file=word_dict)
    data_conll.close()
    word_dict.close()


def create_text():  # 将data.conll文件中的格式修改为：每行一句话,词语之间无空格,起名为data.txt
    data_conll = open('../data/data.conll', 'r', encoding='utf-8')
    data_txt = open('../data/data.txt', 'w', encoding='utf-8')
    for line in data_conll:
        if len(line) > 1:
            print(line.split()[1], end='', file=data_txt)
        else:
            print(file=data_txt)
    data_conll.close()
    data_txt.close()


def forward_max_match():  # 给定词典word.dict,对data.txt进行前向分词,结果输出到data.out中
    word_dict = open('../data/word.dict', 'r', encoding='utf-8')
    data_txt = open('../data/data.txt', 'r', encoding='utf-8')
    data_out = open('../data/data.out', 'w', encoding='utf-8')
    word_set = {word.strip() for word in word_dict}
    m = max(len(word) for word in word_set)
    for line in data_txt:
        line = line.strip()
        sentence = []
        p1 = 0
        while p1 <= len(line) - 1:
            i = m
            p2 = p1 + i
            while p2 > len(line):
                i = i - 1
                p2 = p1 + i
            while True:
                s = line[p1:p2]
                if s not in word_set and i > 1:
                    i = i - 1
                    p2 = p1 + i
                    continue
                else:
                    sentence.append(s)
                    p1 = p2
                    break
        print('\n'.join(sentence), '\n', file=data_out)
    word_dict.close()
    data_txt.close()
    data_out.close()


def backward_max_match():  # 给定词典word.dict,对data.txt进行后向分词,结果输出到data.out中
    word_dict = open('../data/word.dict', 'r', encoding='utf-8')
    data_txt = open('../data/data.txt', 'r', encoding='utf-8')
    data_out = open('../data/data.out', 'w', encoding='utf-8')
    word_set = {word.strip() for word in word_dict}
    m = max(len(word) for word in word_set)
    for line in data_txt:
        line = line.strip()
        sentence = []
        p1 = len(line) - 1
        while p1 >= 0:
            i = m
            p2 = p1 - i
            while p2 < 0:
                i = i - 1
                p2 = p1 - i
            while True:
                s = line[p2:p1 + 1]
                if s not in word_set and i > 0:
                    i = i - 1
                    p2 = p1 - i
                    continue
                else:
                    sentence.append(s)
                    p1 = p2 - 1
                    break
        sentence.reverse()
        print('\n'.join(sentence), '\n', file=data_out)
    word_dict.close()
    data_txt.close()
    data_out.close()


def evaluate():  # 对比data.conll和data.out，给出算法的P/R/F指标
    data_conll = open('../data/data.conll', 'r', encoding='utf-8')
    data_out = open('../data/data.out', 'r', encoding='utf-8')
    answer = [line.split()[1] for line in data_conll if len(line) > 1]
    result = [line.strip() for line in data_out if len(line) > 1]
    sum_answer = len(answer)
    sum_result = len(result)
    correct_result = i = j = 0
    while i < sum_result and j < sum_answer:
        if result[i] == answer[j]:
            correct_result = correct_result + 1
        else:
            str1, str2 = result[i], answer[j]  # 分错词的字符串合并
            while str1 != str2 and i < sum_result - 1 and j < sum_answer - 1:
                if len(str1) > len(str2):
                    j = j + 1
                    str2 = str2 + answer[j]
                else:
                    i = i + 1
                    str1 = str1 + result[i]
        i, j = i + 1, j + 1
    precision = correct_result / sum_result
    recall = correct_result / sum_answer
    f1 = precision * recall * 2 / (precision + recall)
    print('Precision =', correct_result, '/', sum_result, '=', format(precision, '.2%'))
    print('Recall =', correct_result, '/', sum_answer, '=', format(recall, '.2%'))
    print('F-Measure = Precision * Recall * 2 / ( Precision + Recall ) =', format(f1, '.6f'))
    data_conll.close()
    data_out.close()


def main():
    create_dict()  # 构建词典
    create_text()  # 构建毛文本
    forward_max_match()  # 前向最大匹配分词
    print('Evaluation of forward max-match algorithm for chinese word segmentation :')
    evaluate()  # # 前向最大匹配分词算法评价
    backward_max_match()  # 后向最大匹配分词
    print('\nEvaluation of backward max-match algorithm for chinese word segmentation :')
    evaluate()  # 后向最大匹配分词算法评价


if __name__ == '__main__':
    main()
