"""
参数及配置
"""

# main.py
small_dataset = False  # 选择数据集规模
small_train_path = "../data/small_dataset/train.conll"  # 小数据集-训练集
small_dev_path = "../data/small_dataset/dev.conll"  # 小数据集-验证集
big_train_path = "../data/big_dataset/train"  # 大数据集-训练集
big_dev_path = "../data/big_dataset/dev"  # 大数据集-验证集
big_test_path = "../data/big_dataset/test"  # 大数据集-测试集
embedding_path = "../data/embedding/giga.100.txt"  # 预训练好的词向量文件
result_path = "../result/small_dataset_result.txt"  # 保存结果的文件
max_epoch = 50  # 最大的epoch(一次epoch是训练完所有样本一次)
max_no_rise = 10  # epoch连续几轮(验证集最大正确率)没有上升时结束训练

# embedding.py
window = 5  # 上下文窗口大小
embedding_dim = 100  # 每个词向量的维度

# dataloader.py
shuffle = True  # 是否打乱数据集
batch_size = 50  # 多少样本更新一次

# bpnn.py
hidden_layer_size = 354  # 隐藏层中的神经元数量, 2/3(input + output)
activation = 'relu'  # 隐藏层的激活函数
Lambda = 0.01  # L2正则化项系数λ
learning_rate = 0.5  # 初始学习率
embedding_trainable = True  # 是否训练embedding
decay_rate = 0.96  # 学习率衰减速率(防止 loss function 在极小值处不停震荡)
random_seed = 1110  # 随机数种子
