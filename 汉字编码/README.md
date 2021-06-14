# 汉字编码
## 一、目录结构
```
.
|-- README.md
|-- c++
|   |-- data
|   |   |-- gbk_input.txt
|   |   `-- utf-8_input.txt
|   |-- result
|   |   |-- gbk_output.txt
|   |   `-- utf-8_output.txt
|   `-- src
|       `-- chinese_char_seg.cpp
`-- python
    |-- data
    |   |-- gbk_input.txt
    |   `-- utf-8_input.txt
    |-- result
    |   |-- gbk_output.txt
    |   `-- utf-8_output.txt
    `-- src
        `-- chinese-char-seg.py

8 directories, 11 files
```

## 二、运行环境
c++  
codeblocks 20.03  
python 3.8    
PyCharm 2021.1.1 (Professional Edition)
## 三、运行结果
输入：
```
Google是拥有超过10亿用户的公司 也是全球最大的搜索引擎公司
```

输出：
```
G o o g l e 是 拥 有 超 过 1 0 亿 用 户 的 公 司 也 是 全 球 最 大 的 搜 索 引 擎 公 司 
character数量：32
```
