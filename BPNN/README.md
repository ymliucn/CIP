
# BPNN
## 一、目录结构
```
.
|-- README.md
|-- data
|   |-- big_dataset
|   |   |-- dev
|   |   |-- test
|   |   `-- train
|   `-- small_dataset
|       |-- dev.conll
|       `-- train.conll
|-- result
|   |-- small_data_optimized.txt
|   `-- small_data_original.txt
`-- src
    |-- LogLinearModel.py
    |-- config.py
    |-- dataset.py
    |-- main.py
    `-- utils.py

5 directories, 13 files
```

## 二、运行环境
### (1)小数据集
windows10    
anaconda3 python 3.8       
PyCharm 2021.1.2 (Professional Edition)
### (2)大数据集
linux    
anaconda3 python 3.8
```
srceen -s BPNN
python3.8 main.py
```
## 三、参数及解释
 参数 | 解释 | 默认值 |
 :-----: | :-----: | :-----: |
 small_dataset | 选择数据集 | True
 max_epoch | 最大的epoch | 50
 max_no_rise | 验证集最大正确率连续几轮没有上升时结束训练 | 10
 window | 上下文窗口大小| 5
 embedding_dim | 每个词向量的维度 | 100
 shuffle | 训练时是否打乱数据集 | True 
 batch_size | 批量大小 | 50 
 hidden_layer_size | 隐藏层中的神经元数量 | 354
 activation| 隐藏层的激活函数 | 'relu'
 Lambda | L2正则化项系数λ | 0.01
 learning_rate | 初始学习率 | 0.5
 embedding_trainable | 是否训练embedding | True
 decay_rate | 学习率衰减率 | 0.96
 random_seed | 随机数种子 | 1110

## 四、运行结果
### (1)小数据集
small_dataset | 迭代轮数 | 验证集最大正确率 | 总共耗时 |
 :-----: | :-----: | :-----: | :-----: |
 True | 19/29 | 89.31% | 0:16:36.302705
 
### (2)大数据集
 small_dataset | 迭代轮数 | 验证集最大正确率 | 测试集正确率 | 总共耗时 |
 :-----: | :-----: | :-----: | :-----: | :-----: |
 False | 23/33 | 94.18% | 93.69% | 3:43:21.732549

