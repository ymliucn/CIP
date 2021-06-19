
# HMM词性标注
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
`-- src
    `-- hmm_viterbi.py

4 directories, 7 files
```

## 二、运行环境
anaconda3 python 3.8       
PyCharm 2021.1.2 (Professional Edition)
## 三、运行结果
 训练集  | 测试集 | alpha | 标注正确词数 | 总词数 | 正确率 | 总共耗时
 :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
 train.conll  | dev.conll | 0.3 | 38112 | 50319 | 75.74% | 0:00:03.630253
 train  | dev | 0.01 | 18137 | 20454 | 88.67% | 0:00:16.261467
 train  | test | 0.01 | 44355 | 50319 | 88.15% | 0:00:36.579420

