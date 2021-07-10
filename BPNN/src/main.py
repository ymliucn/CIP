from config import *
from bpnn import BPNN
from dataset import Dataset
from datetime import datetime
from embedding import Embedding
from dataloader import DataLoader


def main():
    start_time = datetime.now()

    with open(result_path, 'a', encoding='utf-8') as result:  # 读入数据
        result.write("一、读入数据\n")
    if small_dataset:  # 选择数据集
        train_set = Dataset(small_train_path)
        dev_set = Dataset(small_dev_path)
        test_set = None
    else:
        train_set = Dataset(big_train_path)
        dev_set = Dataset(big_dev_path)
        test_set = Dataset(big_test_path)

    with open(result_path, 'a', encoding='utf-8') as result:  # 进行词嵌入,增加上下文窗口
        result.write("\n二、Embedding\n")
    embedding = Embedding(embedding_path)
    embedding.extend_embedding(train_set)
    unique_tag_num, tag2id = train_set.unique_tag_num, train_set.tag2id
    train_data = embedding.add_context_window(train_set, unique_tag_num, tag2id)
    dev_data = embedding.add_context_window(dev_set, unique_tag_num, tag2id)
    test_data = embedding.add_context_window(test_set, unique_tag_num, tag2id) if test_set else None

    n = len(train_data)  # 训练集的大小
    max_accuracy = 0  # 最大正确率
    best_epoch = 0  # 最佳模型的epoch
    global_step = 100000  # 学习率衰减最大步数
    output_layer_size = train_set.unique_tag_num  # 输出层神经元数目(即词性数目)
    bpnn = BPNN(embedding, output_layer_size)
    lr = bpnn.learning_rate
    with open(result_path, 'a', encoding='utf-8') as result:
        result.write("\n三、BP神经网络开始训练\n")  # 开始训练
    for epoch in range(1, max_epoch + 1):
        dataloader = DataLoader(train_data)
        for step, batch in enumerate(dataloader.batch_list):
            bpnn.learning_rate = lr * decay_rate ** (step / global_step)  # 学习率按步长衰减
            bpnn.weight_decay = 1 - bpnn.learning_rate * bpnn.Lambda / n  # 更新权重衰减
            bpnn.fit(batch)

        with open(result_path, 'a', encoding='utf-8') as result:
            result.write(f'\nepoch {epoch} 训练完成\n')
        train_correct, train_all, train_accuracy = bpnn.evaluate(train_data)
        dev_correct, dev_all, dev_accuracy = bpnn.evaluate(dev_data)
        with open(result_path, 'a', encoding='utf-8') as result:
            result.write(f'训练集{train_set.file_name}：Accuracy = {train_correct} / {train_all} = {train_accuracy}\n')
            result.write(f'验证集{dev_set.file_name}：Accuracy = {dev_correct} / {dev_all} = {dev_accuracy}\n')
        if test_set:
            test_correct, test_all, test_accuracy = bpnn.evaluate(test_data)
            with open(result_path, 'a', encoding='utf-8') as result:
                result.write(f'测试集{test_set.file_name}：Accuracy = {test_correct} / {test_all} = {test_accuracy}\n')

        if dev_accuracy > max_accuracy:  # 更新最大accuracy和出现最大accuracy的epoch
            best_epoch = epoch
            max_accuracy = dev_accuracy

        if epoch - best_epoch >= max_no_rise:
            with open(result_path, 'a', encoding='utf-8') as result:
                result.write(f'\n验证集正确率连续{max_no_rise}轮没有上升,BP神经网络结束训练。')
            break
    with open(result_path, 'a', encoding='utf-8') as result:
        result.write(f'\n最大正确率：epoch：{best_epoch},  Max Accuracy：{max_accuracy}\n')
    end_time = datetime.now()
    with open(result_path, 'a', encoding='utf-8') as result:
        result.write(f'\n四、总共耗时：{end_time - start_time}')


if __name__ == '__main__':
    main()
