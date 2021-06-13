def utf8_split_character():
    f_in = open('../data/utf-8_input.txt', 'r', encoding='utf-8')
    s = f_in.read().split()
    num = 0
    print('utf-8编码分字结果：')
    for t in s:
        for c in t:
            print(c, end=' ')
            num = num + 1
    print('\ncharacter数量：' + str(num) + '\n')
    f_in.close()
    f_out = open('../result/utf-8_output.txt', 'w', encoding='utf-8')
    for t in s:
        for c in t:
            print(c, end=' ', file=f_out)
    print('\ncharacter数量：' + str(num), file=f_out)
    f_out.close()


def gbk_split_character():
    f_in = open('../data/gbk_input.txt', 'r')
    s = f_in.read().split()
    num = 0
    print('gbk编码分字结果：')
    for t in s:
        for c in t:
            print(c, end=' ')
            num = num + 1
    print('\ncharacter数量：' + str(num))
    f_in.close()
    f_out = open('../result/gbk_output.txt', 'w')
    for t in s:
        for c in t:
            print(c, end=' ', file=f_out)
    print('\ncharacter数量：' + str(num), file=f_out)
    f_out.close()


def main():
    utf8_split_character()
    gbk_split_character()


if __name__ == "__main__":
    main()
