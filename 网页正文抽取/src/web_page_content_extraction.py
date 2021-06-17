from bs4 import BeautifulSoup


def crawler(html_path, txt_path):
    html = open(html_path, encoding='utf-8')
    soup = BeautifulSoup(html.read(), "lxml")
    html.close()
    txt = open(txt_path, 'w')
    print('title:\n' + soup.title.string, file=txt)  # title标签内容
    print('body:', file=txt)
    s = list(''.join(soup.body.strings).strip())  # 所有正文内容字符列表
    n = i = 0
    line = ''
    while i < len(s):
        if s[i] != '\n':
            line += s[i]
            i += 1
        else:
            while s[i] == '\n':
                n += 1
                del s[i]
            line = line.strip()
            if n >= 2:  # 连续的空行压缩为一个空行
                line += '\n'
            print(line, file=txt)
            line = ''
            n = 0
    print(line.strip(), file=txt)
    print('link:', file=txt)
    for a in soup.find_all('a'):
        print(a.string, a['href'], file=txt)  # a标签内容,a标签href(超链接)属性
    txt.close()


def main():
    crawler('../data/1.html', '../result/1.txt')
    crawler('../data/2.html', '../result/2.txt')


if __name__ == '__main__':
    main()
