# encoding='utf-8'


def compare(txt, writePath):
    '''
    判断文本文件中图片路径是否有重复
    :return:
    '''

    outfiile = open(writePath, 'a+', encoding='utf-8')
    f = open(txt, 'r', encoding='utf-8')
    lines_seen = set()

    i = 1
    for line in f:
        #file,label = line.split()
        #if file not in lines_seen:
        if line not in lines_seen:
            outfiile.write(line)
            lines_seen.add(line)
            if i % 1000 == 0:
                print("已经处理行数：", i)
                i += 1



def count_char(writePath, count_path):
    f = open(writePath, 'r', encoding='utf-8')
    s = ""
    for line in f.readlines():
        print("line:", line)
        file, label = line.split()
        label = label.replace("\n", "")
        s = s + label

    resoult = {}
    for i in s:
        resoult[i] = s.count(i)
    print(resoult)

    # 排序
    r_sort = sorted(resoult.items(), key=lambda x: x[1], reverse=True)
    print(r_sort)

    with open(count_path, "w", encoding="utf-8") as f1:
        for l in r_sort:
            list1 = list(l)
            str1 = list1[0] + ' ' + str(list1[1])
            f1.write(str1 + "\n")
            #f1.write(str(l) + "\n")


def test():
    # 统计字符个数
    str = "苏E85QE8鄂AH72W2赣HA0696"
    resoult = {}
    for i in str:
        resoult[i] = str.count(i)
    print(resoult)



if __name__ == "__main__":
    txt = 'data/ocr/train.txt'
    writePath = 'data/ocr/train_1.txt'
    count_path = "data/ocr/counts.txt"

    # txt = 'data/test.txt'
    # writePath = 'data/test_1.txt'
    # count_path = "data/counts.txt"


    # 删除重复行
    compare(txt, writePath)

    # 统计各个字符个数
    count_char(writePath, count_path)
