
# 最大匹配分词
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
anaconda3 python 3.8       
PyCharm 2021.1.2 (Professional Edition)
## 三、运行结果
```
Evaluation of forward max-match algorithm for chinese word segmentation :
Precision = 20263 / 20397 = 99.34%
Recall = 20263 / 20454 = 99.07%
F-Measure = Precision * Recall * 2 / ( Precision + Recall ) = 0.992044

Evaluation of backward max-match algorithm for chinese word segmentation :
Precision = 20273 / 20404 = 99.36%
Recall = 20273 / 20454 = 99.12%
F-Measure = Precision * Recall * 2 / ( Precision + Recall ) = 0.992364
```

