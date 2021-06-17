
# 网页正文抽取
## 一、目录结构
```
.
|-- README.md
|-- data
|   |-- 1.html
|   |-- 1_files
|   |   |-- style.css
|   |   `-- zhenghua-2018.jpg
|   `-- 2.html
|-- result
|   |-- 1.txt
|   `-- 2.txt
`-- src
    `-- web_page_content_extraction.py

4 directories, 8 files

```

## 二、运行环境
anaconda3 python 3.8       
PyCharm 2021.1.1 (Professional Edition)
## 三、运行结果
1.txt
```
title:
Zheng-Hua Li
body:
李正华 (English Homepage)
Email: prefix@suffix, where prefix=``zhli13'' and suffix=``suda dot edu dot cn'' || prefix=``zhenghua-nlp'' and suffix=``qq dot com''

简介
目前我在苏州大学 计算机科学与技术学院 人类语言技术研究所(HLT) 担任副教授（自2016年7月），硕士生导师（自2017年12月）。
2006、2008、2013年取得哈尔滨工业大学学士、硕士、博士学位；2010年在新加坡资讯通信研究院访学半年；2013年8月加入苏州大学。从2005年（大三）开始接触自然语言处理（Natural Language Processing, NLP）和人工智能（Artifical Intelligence, AI），至今涉及的研究工作包括词法、句法、语义等方向，做出了一些成果：

在顶级国际期刊和会议（TASLP/ACL/EMNLP/NAACL/COLING/AAAI/IJCAI）上发表学术论文~20篇
承担国家自然科学基金青年项目1项、面上项目1项；先后与百度、腾讯、阿里、华为等公司科研合作
持续建设并维护数据标注平台（DAP）、汉语理解平台（CUP）
曾于2008-2011年开发并维护哈工大LTP平台
持续标注高质量句法语义数据，目前汉语依存句法数据~12万句（CODT）
曾于2010年主持二次标注哈工大6万句依存句法树库

我近期的研究兴趣包括：

分词、多粒度分词
句法分析
语义分析，如SRL（语义角色标注、又称浅层语义分析）、AMR等
数据标注方法、语料库构建、语言资源
无监督学习方法、领域移植问题
噪音文本规范化，或文本错误纠正（文本纠错）（希望可以打造一个面向汉语的产品，类似grammarly）

我的长远职业规划和目标是：

认真上课，成为一名优秀的教师，2022年之前争取写1本跟所授课程相关的教材或讲义（Python/Linux）
认真指导研究生，培养出优秀的硕士和博士研究生
做出具有重大影响力的研究成果，促进NLP和AI领域的发展

我的座右铭：人生苦短，珍惜时间和精力，尽量只做自己想做的，并全力做好。
我的一些思考
NEWS（小组新闻）

教学

2021春
Linux操作系统 (网页，内含)

2020秋
Python程序设计 (网页，内含笔记、板书、讲课视频)

2020春
Linux操作系统 (网页，重新整理了2018春的视频和笔记）

2019秋
Linux操作系统 (文正学院9人，网页，内含笔记、板书)

2019春
Linux操作系统 (网页，内含笔记、板书)

2019春
信息检索课程设计(网页，内含课件和作业)

2018秋
Python程序设计 (网页)

2018春
Linux操作系统 (网页，内含笔记、和视频百度云链接)

2017秋
Python程序设计 (网页，录播课，建议看2020秋Python视频和笔记)

2016秋
Python程序设计(网页，内含PPT)

2016春
信息检索课程设计 (网页，内含作业和数据)

2015秋
中文信息处理 （网页，内含讲义、作业和数据等，推荐看我主页中的新生编程基础练习）

2015秋
Linux操作系统 (网页，内含PPT)

科研成果演示系统
苏州大学汉语理解平台 (CUP)：输入一个句子，输出分词、词性标注、依存句法分析结果。（蒋炜同学维护）

苏州大学数据标注平台 (SUDAP)：给定句子，人工标注分词、句法、命名实体等信息（陆凯华、沈嘉钰同学开发并维护）

苏州大学多粒度分词系统 ：输入一个句子，给出不同粒度的分词信息，以树状形式展示（蒋炜同学开发并维护）


正在进行的科研项目
汉语开放依存树库CODT：雇用几十名本科同学，长期兼职标注句子的句法信息

汉语开放谓词论元数据集COPAD：雇用几十名本科同学，长期兼职标注句子的语义信息


在读学生 (字母序)【求真、务实、独立、自由】
招生：李正华的招生说明
Suda-HLT-LAGroup学生管理规则
Suda-HLT-LAGroup本科俱乐部同学管理规则
NLP入门基础编程训练
2017级博士(1)：李英（昆明理工考博）

2018级博士(2)：龚晨（苏大保研16硕、直博）、夏庆荣（苏大保研16硕、直博）

2020级博士(1)：刘亚慧（山东农大考研18硕，直博)

2018级硕士(5)：蒋炜（苏大）、陆凯华（苏大、导师张民、专硕）、吴锟（浙江理工）、张宇（苏大）、刘亚慧（山东农大，直博）

2019级硕士(4)：沈嘉钰（苏大）、杨浩苹（苏大保研）、周厚全（矿大保研）、周明月（苏大保研）

2020级硕士(6)：侯洋（苏大保研）、李嘉诚（燕山大学、专硕）、李帅克（苏大保研）、李扬（苏大、导师张民）、刘泽洋（华北电力、专硕、导师张民）、周仕林（苏大、专硕）

2021级硕士(6)：崔秀莲（苏大保研）、黄赛豪（苏大保研）、章岳（苏大保研）、考研3位通过初试

2022级硕士(4)：上两级同学比较多，22级计划招4位，如有意愿，请提早联系我。


毕业学生 (字母序)
2017级硕士(4)：黄德朋（苏科技，华为杭州实习，小红书上海）、江心舟（苏大保研，百度北京实习）、彭雪（山东农大，华为杭州实习、移动苏州）、章波（苏大，阿里巴巴杭州达摩院实习、转正）

2016级硕士(5)：郭丽娟（江西财经保研，科沃斯实习，狗尾草公司工作）、孙佳伟（北航，导师张民，搜狗北京实习，微软苏州工作）、朱运（山西大学，搜狗北京工作）、龚晨（苏大保研、直博）、夏庆荣（苏大保研、直博）

2015级硕士(3)：陈伟（南阳理工、爱奇艺北京实习、爱奇艺北京工作）、凡子威（滁州学院、科大讯飞北京实习、搜狗北京工作） 、张月（苏大保研、阿里巴巴杭州实习、阿里巴巴杭州工作）

2014级硕士(1)：巢佳媛（苏大、微软北京实习、阿里巴巴杭州工作）

本科毕业论文指导：指导详情网页
请看毕业设计指导详情网页，包括pdf，ppt

2017级本科毕设(6)：崔秀莲、窦晨晖、黄赛豪、司英杰、严福康、章岳 [2021春]

2016级本科毕设(6)：侯洋、黎霞、李帅克、周仕林、韩欣艳、杨奕 [2020春]

2015级本科毕设(7)：陈婷、李烨秋、沈嘉钰、杨浩苹、袁源、周明月、周厚全（矿大）[2019春]

2014级本科毕设(3)：蒋炜、李丹、陆凯华 [2018春]

2013级本科毕设(5)：胡蝶、江心舟、孙文杰、严秋怡、章波 [2017春]

2012级本科毕设(2)：龚晨、夏庆荣 [2016春]

2011级本科毕设(4)：陆芳丽、穆景泉、王效静、张月 [2015春]

2010级本科毕设(1)：郁俊杰 [2014春]


最新报告和论文：链接
老的中文论文：
每位硕士同学通常必须发表一篇较高质量的中文论文：1) 锻炼科研、实验、写作等能力；2) 硕士毕业论文做准备；3) 达到学院毕业标准

吴锟投出一篇论文，被CCKS录用，推荐核心期刊

陆凯华投出一篇论文，已被厦门大学学报录用

彭雪, 李正华, 张民. 2020.4. 基于语言模型微调的跨领域依存句法分析. 计算机应用与软件（已录用）. 2022?. (pdf)

章岳, 黄赛豪, 陆凯华, 李正华. 2020.1. 基于模板的中文上下位关系抽取方法. 计算机应用与软件（已录用）. 2022?. (pdf)

刘亚慧, 杨浩苹, 李正华, 张民. 2020. 一种轻量级的语义角色标注规范. 中文信息学报. 2020, 34(4):10-20  (pdf)

李正华, 张民. 2019. 自然语言处理的光明未来---ACL 2019参会总结和感想 . 中国计算机学会通讯. 2019, 15 (11): 81-83
(pdf)

蒋炜, 李正华, 张民. 2019. Syntax-enhanced UCCA Semantic Parsing (句法增强的UCCA语义分析). Proceedings of the 8th International Conference on Natural Language Processing and Chinese Computing (NLPCC-2019), Dunhuang（敦煌）, 中国, 13-14 Oct., 2019. 推荐发表至：北京大学学报（自然科学版）
(pdf)

黄德朋, 李正华, 龚晨, 张民. 2019. Neural Network Coupled Model for Conversion and Exploitation of Heterogeneous Lexical Annotations (基于神经耦合模型的异构词法数据转化和融合). Proceedings of the 8th International Conference on Natural Language Processing and Chinese Computing (NLPCC-2019), Dunhuang（敦煌）, 中国, 13-14 Oct., 2019. 推荐发表至：北京大学学报（自然科学版）
(pdf)

朱运, 李正华, 黄德朋, 张民. 2019年9月20日. 基于弱标注数据的汉语分词领域移植. 中文信息学报. 2019, 33 (9): 1-8
(pdf)

凡子威, 张民, 李正华. 2019. 基于BiLSTM并结合自注意力机制和句法信息的隐式篇章关系分类篇章分析. 计算机科学. 2019, 46(5):214-220
(pdf)

郭丽娟, 彭雪, 李正华, 张民. 2019. 面向多领域多来源文本的汉语依存句法树库构建. 中文信息学报. 2019, 33(2):34-42
(pdf)

郭丽娟, 李正华, 彭雪, 张民. 2018. 适应多领域多来源文本的汉语依存句法数据标注规范. 中文信息学报. 2018, 32(10):28-35-52
(pdf)

孙佳伟, 李正华, 陈文亮, 张民. 2018. Hypernym Relation Classification based on Word Pattern (基于词模式嵌入的词语上下位关系分类). Proceedings of the 7th International Conference on Natural Language Processing and Chinese Computing (NLPCC-2018), Hohhot（呼和浩特）, 中国, 26-30 Aug, 2018.
推荐发表至：北京大学学报（自然科学版）. 2019, 55(1):1-7
(pdf)

高恩婷, 巢佳媛, 李正华. 2015. 面向词性标注的多资源转化研究. 北京大学学报(自然科学版). 2015, 51(2):328-334 (NLPCC 2014优秀论文转投)
(pdf)

李正华, 李渝勤, 刘挺, 车万翔. 2013. 数据驱动的依存句法分析方法研究. 智能计算机与应用. 2013, 3(5):1-4
(pdf)

李正华. 2013. 汉语依存句法分析关键技术研究. 博士学位论文. 2013年3月. 哈尔滨工业大学. (pdf, pptx)

刘挺, 车万翔, 李正华. 2011. 语言技术平台. 中文信息处理. 2011, 25(6):53-62
(pdf)

李正华, 车万翔, 刘挺. 2010. 基于柱搜索的高阶依存句法分析. 中文信息处理. 2010, 24(1):37-41
(pdf)

李正华, 车万翔, 刘挺. 2008. 短语结构树库向依存结构树库转化研究. 中文信息处理. 2008, 22(6):14-19 (第四届全国学生计算语言学研讨会优秀论文转投)
(pdf)

郎君, 秦兵, 刘挺, 李正华, 李生. 2008. 中文人称名词短语单复数自动识别. 自动化学报. 2008, 34(8):972-979
(pdf)

最后更新(UTC/GMT): 04/21/2021 11:34:53
link:
English Homepage http://hlt.suda.edu.cn/~zhli/en.html
苏州大学 http://www.suda.edu.cn/
计算机科学与技术学院 http://scst.suda.edu.cn/
HLT http://hlt.suda.edu.cn/
我的一些思考 http://hlt.suda.edu.cn/index.php/My-thoughts-zhenghua
NEWS（小组新闻） http://hlt.suda.edu.cn/index.php/News-zhenghua
网页 http://hlt.suda.edu.cn/index.php/linux-2021-spring
网页 http://hlt.suda.edu.cn/index.php/python-2020-fall
网页 http://hlt.suda.edu.cn/index.php/linux-2020-spring-zhenghua
网页 http://hlt.suda.edu.cn/index.php/linux-2019-fall
网页 http://hlt.suda.edu.cn/index.php/linux-2019-spring
网页 http://hlt.suda.edu.cn/index.php/ir-2019-spring
网页 http://hlt.suda.edu.cn/index.php/python-2018
网页 http://hlt.suda.edu.cn/index.php/linux-2018-spring
网页 http://hlt.suda.edu.cn/gongchen/teach/python-2017-fall
网页 http://hlt.suda.edu.cn/~zhli/teach/python-2016-fall
网页 http://hlt.suda.edu.cn/~zhli/teach/ir-2016-spring
网页 http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall
网页 http://hlt.suda.edu.cn/~zhli/teach/linux-2015-fall
苏州大学汉语理解平台 (CUP) http://hlt-la.suda.edu.cn/
苏州大学数据标注平台 (SUDAP) http://139.224.234.18/anno-sys
苏州大学多粒度分词系统  http://139.224.234.18:5000/demo
汉语开放依存树库CODT http://hlt.suda.edu.cn/index.php/CODT
汉语开放谓词论元数据集COPAD http://hlt.suda.edu.cn/index.php/COPAD
李正华的招生说明 http://hlt.suda.edu.cn/index.php/Zhenghua-recruiting
Suda-HLT-LAGroup学生管理规则 http://hlt.suda.edu.cn/~zhli/LAGroupRules
Suda-HLT-LAGroup本科俱乐部同学管理规则 http://hlt.suda.edu.cn/index.php/Lagroup-club-rules
NLP入门基础编程训练 http://hlt.suda.edu.cn/index.php/New-stu-training
龚晨 http://hlt.suda.edu.cn/gongchen
夏庆荣 http://hlt.suda.edu.cn/kiro
龚晨 http://hlt.suda.edu.cn/gongchen
夏庆荣 http://hlt.suda.edu.cn/kiro
凡子威 http://hlt.suda.edu.cn/zwfan
张月 http://hlt.suda.edu.cn/yzhang
指导详情网页 http://hlt.suda.edu.cn/index.php/zhenghua-undergraduate-thesis
链接 http://hlt.suda.edu.cn/index.php/la-paper-report-talk-etc
pdf http://hlt.suda.edu.cn/~zhli/papers/xx.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/xx.pdf
pdf http://hlt.suda.edu.cn/index.php/%E6%96%87%E4%BB%B6:%E4%B8%80%E7%A7%8D%E8%BD%BB%E9%87%8F%E7%BA%A7%E7%9A%84%E6%B1%89%E8%AF%AD%E8%AF%AD%E4%B9%89%E8%A7%92%E8%89%B2%E6%A0%87%E6%B3%A8%E8%A7%84%E8%8C%83.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/ACL2019-attending.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/jiangwei-peking20-ucca.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/depeng-peking20-coupled.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/zhuyun-cip19-wordseg.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/ziwei-cs19-discourse.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/lijuan-cip19-treebank.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/lijuan-jocip18-guideline.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/jiawei-peking18-hypernym.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/zhenghua-peking15-pos.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/zhenghua-13-dp.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/zhenghua-2013-phd-thesis.pdf
pptx http://hlt.suda.edu.cn/~zhli/talks/zhenghua-2013-phd-thesis-defence.pptx
pdf http://hlt.suda.edu.cn/~zhli/papers/zhenghua-jocip11-ltp.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/zhenghua-jocip10-beam.pdf
优秀论文 http://hlt.suda.edu.cn/~zhli/papers/zhenghua-swcl08-treebanks.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/zhenghua-jocip08-treebanks.pdf
pdf http://hlt.suda.edu.cn/~zhli/papers/langjun-08-noun-number.pdf

```
2.txt
```
title:
Teaching by Wenliang
body:
信息检索课程设计（Information Retrieval）Course Resources
教师: 陈文亮
助教: 郁俊杰
2016春季学期，计算机学院大一本科生，选修课
上机时间：周一15:10-17:00；地点：理工楼243

编程作业提交细则 (ppt)：重要！内含反作弊声明。请仔细阅读。
只在上课现场收作业，请勿通过邮件提交。

Course 1 (2.29)
课程介绍及评分规则 (ppt)
Assignment 1 (word-count; 10分; 完成时间：2.29-3.7, 最迟3.14上机课检查，过期不侯)
统计单词频率 (ppt)
数据下载 (sample-en.txt)
Assignment 2 (word-seg; 15分; 完成时间：3.14-3.28, 最迟4.11上机课检查，过期不侯)
中文分词：前向最大匹配 (ppt)
数据下载 (dict)
数据下载 (sentences)
数据下载 (answers)

Last modified at (UTC/GMT):
link:
ppt http://hlt.suda.edu.cn/~wlchen/ir2016/rules.ppt
ppt http://hlt.suda.edu.cn/~wlchen/ir2016/L0.Introduction.ppt
ppt http://hlt.suda.edu.cn/~wlchen/ir2016/assignment-1-word-count/E1.ppt
sample-en.txt http://hlt.suda.edu.cn/~wlchen/ir2016/assignment-1-word-count/d1.txt
ppt http://hlt.suda.edu.cn/~wlchen/ir2016/assignment-2-word-seg/wordseg.ppt
dict http://hlt.suda.edu.cn/~wlchen/ir2016/assignment-2-word-seg/data/corpus.dict.txt
sentences http://hlt.suda.edu.cn/~wlchen/ir2016/assignment-2-word-seg/data/corpus.sentence.txt
answers http://hlt.suda.edu.cn/~wlchen/ir2016/assignment-2-word-seg/data/corpus.answer.txt

```

