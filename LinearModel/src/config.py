small_train_path = "../data/small_dataset/train.conll"  # 小数据集-训练集
small_dev_path = "../data/small_dataset/dev.conll"  # 小数据集-验证集
big_train_path = "../data/big_dataset/train"  # 大数据集-训练集
big_dev_path = "../data/big_dataset/dev"  # 大数据集-验证集
big_test_path = "../data/big_dataset/test"  # 大数据集-测试集
result_path = "../result/small_data_w.txt"  # 结果文件的路径

small_dataset = True  # 选择数据集规模
shuffle = True  # 是否打乱数据集
max_iterations = 100  # 最大迭代轮数
max_no_rise = 10  # 验证集最大正确率连续几轮没有上升时结束训练

averaged_perceptron = False  # 是否使用权重累加, True使用v, False使用w
