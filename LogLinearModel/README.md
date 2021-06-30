
# LogLinearModel
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
<!-- 
linux    
anaconda3 python 3.8
```
srceen -s LinearModel
python3.8 mian.py
```
-->
## 三、运行结果
### (1)小数据集
 步长衰减率  | 正则项权重  | 初始η  | batch size | shuffle | 迭代轮数 | 验证集最大正确率 | 总共耗时 |
 :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
 ×  | × | 0.5 | 50 | √ | 10/15 | 87.49% | 0:04:43.670394
 0.96  | 0.0001 | 0.5 | 50 | √ | 10/15 | 87.54% | 0:04:41.559895
 
### (2)大数据集
 步长衰减率  | 正则项权重 | 初始η  | batch size | shuffle | 迭代轮数 | 验证集最大正确率 | 测试集正确率 | 总共耗时 |
 :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
<!--  ×  | × | 0.5 | 50 | √ | 10/15 | 87.49% | 87.49% | 0:04:43.670394
 √  | √ | 0.5 | 50 | √ | 10/15 | 87.54% | 87.54% | 0:04:41.559895 -->
