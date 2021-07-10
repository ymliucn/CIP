from config import *
from dataset import Dataset
from datetime import datetime
from CRF import CRF


def main():
    start_time = datetime.now()

    print("一、读入数据")
    if small_dataset:  # 选择数据集
        train_set = Dataset(small_train_path)
        dev_set = Dataset(small_dev_path)
        test_set = None
    else:
        train_set = Dataset(big_train_path)
        dev_set = Dataset(big_dev_path)
        test_set = Dataset(big_test_path)

    crf = CRF()
    crf.fit(train_set, dev_set, test_set)

    end_time = datetime.now()
    print(f'\n四、总共耗时：{end_time - start_time}')


if __name__ == '__main__':
    main()
