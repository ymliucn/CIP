
# LinearModel
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
|   |-- small_data_v.txt
|   `-- small_data_w.txt
`-- src
    |-- LinearModel.py
    |-- config.py
    |-- dataset.py
    `-- main.py

5 directories, 12 files
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
srceen -s LinearModel
python3.8 mian.py
```
## 三、运行结果
### (1)小数据集
 Averaged Perceptron  | shuffle | 验证集最大正确率 | 总共耗时 |
 :-----: | :-----: | :-----: | :-----: |
 False  | True | 85.79% | 0:03:39.260136
 True  | True | 86.08% | 0:04:42.280242
 
### (2)大数据集
 Averaged Perceptron  | shuffle | 验证集最大正确率 | 测试集正确率 | 总共耗时 |
 :-----: | :-----: | :-----: | :-----: | :-----: |
<!--  False  | True | 85.79% | 85.79% | 0:03:39.260136
 True  | True | 86.08% | 86.08% | 0:04:42.280242 -->
