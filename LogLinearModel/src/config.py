small_train_path = "../data/small_dataset/train.conll"  # 小数据集-训练集
small_dev_path = "../data/small_dataset/dev.conll"  # 小数据集-验证集
big_train_path = "../data/big_dataset/train"  # 大数据集-训练集
big_dev_path = "../data/big_dataset/dev"  # 大数据集-验证集
big_test_path = "../data/big_dataset/test"  # 大数据集-测试集
result_path = "../result/small_data_optimized.txt"  # 结果文件的路径

max_iterations = 50  # 最大迭代轮数
max_no_rise = 5  # 验证集最大正确率连续几轮没有上升时结束训练
shuffle = True  # 是否打乱数据集
batch_size = 50  # 批量大小
init_eta = 0.5  # 初始更新步长
decay_rate = 0.96  # 步长衰减速率
C = 0.0001  # 正则化项权重

small_dataset = True  # 选择数据集规模
