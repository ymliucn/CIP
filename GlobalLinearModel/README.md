
# GlobalLinearModel
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
    |-- GlobalLinearModel.py
    |-- config.py
    |-- dataset.py
    `-- main.py

5 directories, 12 files
```

## 二、运行环境
### (1)小数据集
windows10    
anaconda3 python 3.8       
PyCharm 2021.1.3 (Professional Edition)
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
 Averaged Perceptron  | shuffle | 迭代轮数 | 验证集最大正确率 | 总共耗时 |
 :-----: | :-----: | :-----: | :-----: | :-----: |
 ×  | √ | 13/18 | 87.73% | 0:04:56.323676
 √  | √ | 15/20 | 88.21% | 0:05:42.915236
 
### (2)大数据集
 Averaged Perceptron  | shuffle | 迭代轮数 | 验证集最大正确率 | 测试集正确率 | 总共耗时 |
 :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
<!--  ×  | √ | 13/18 | 87.73% | 87.73% | 0:04:56.323676
 √  | √ | 15/20 | 88.21% | 88.21% | 0:05:42.915236 -->
