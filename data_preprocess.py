import os
import xml.etree.ElementTree as ET
import random
from base_tagger import *
import re
# root = r"data/postagged"
# # root = r'D:\重要的文件夹\ctb8.0\data\postagged'
# file_path = os.listdir(root)
# print(file_path)
# f = open(root + '\\' + 'chtb_0930.nw.pos', encoding='utf-8')
sid = 0


def parseTagged(line, fp=None, i=0,j=0):
    old_line = line
    line = line.strip()
    line = line.split(' ')
    s = []
    t = []
    special_pos = {'NR-SHORT','NN-SHORT','NT-SHORT'}
    try:
        for j in range(len(line)):
            p1,p2 = line[j].split('_')
            # p1, p2 = re.split('',line[j])
            # if p2 in special_pos:
            #     print(p1+p2)
            #     print(old_line)
            # if (p1[0]=='我' or p1[0]=='你') and p2 =='NN':
            #     print(line)
            # if p1=='':
            #     print(old_line)
            # if p2 in {'rs','nd','nrf','dp'}:
            #     print(fp,line[j])
            #     print(old_line)

            # if p2 == 'nr' and len(p1)==1:
            #     print(line[j-1],line[j],line[j+1])
            #
            # if p1 == 'ｕＵＴｘt.cＯｍ':
            #     print(fp)
            p2 = p2.lower()
            # if len(p1)==1 and p2 in {'y','t','n','b','r','v','d','a','v','a','n','m'}:
            #     print(j)
            #     print(line)
            #     print()
            # if p2 in {'an','vn','l'}:
            #     print(p1+p2)
            s.append(p1)
            t.append(p2)
            # if '-' in p2:
            #     print(line)
    except ValueError as e:
        if fp:
            print(fp)
            print(i)
            print(j)
            print(line)
            print(e)
    return s, t


def loadCTB8Data(root):
    raw = []
    tag = []
    segmented = []
    file_path = os.listdir(root)
    for fp in file_path:
        if fp < r"chtb_1152":
            f = open(root + '/' + fp, encoding='utf-8')
            lines = f.readlines()

            for i in range(len(lines)):
                if lines[i][0:4] == '</S>':
                    line = lines[i - 1]
                    if line[0] == '<':
                        continue
                    s, t = parseTagged(line, fp, i)
                    segmented.append(s)
                    tag.append(t)
                    raw.append(''.join(s))
        elif fp < r'chtb_3146':
            f = open(root + '/' + fp, encoding='utf-8')
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i][0] != '<':
                    s, t = parseTagged(lines[i], fp, i)
                    segmented.append(s)
                    tag.append(t)
                    raw.append(''.join(s))
        else:
            f = open(root + '/' + fp, encoding='utf-8')
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i][0] != '<':
                    if lines[i][0] == '\n':
                        continue
                    s, t = parseTagged(lines[i], fp, i)
                    segmented.append(s)
                    tag.append(t)
                    raw.append(''.join(s))

    return segmented, tag, raw


def loadCTB3Data(root):
    train_raw = []
    train_segmented = []
    train_tag = []
    dev_raw = []
    dev_segmented = []
    dev_tag = []
    test_raw = []
    test_segmented = []
    test_tag = []
    file_path = os.listdir(root)
    # load data that distributes like used in paper
    for fp in file_path:
        if (fp<r'chtb_0326'and fp>r'chtb_0301'):
            f = open(root + '/' + fp, encoding='utf-8')
            lines = f.readlines()
            # print(len(lines))
            for i in range(len(lines)):
                # print(lines[i][0:4])
                if lines[i][0:4] == '</S>':
                    # print(1)
                    line = lines[i - 1]
                    if line[0] == '<':
                        continue
                    s, t = parseTagged(line, fp, i)
                    dev_segmented.append(s)
                    dev_tag.append(t)
                    dev_raw.append(''.join(s))
                    # print('dev:',len(dev_segmented))
                    # print('dev:',len(dev_tag))
                    # print('dev:',len(dev_raw))

        elif (fp<r'chtb_0301'and fp>r'chtb_0271'):
            f = open(root + '/' + fp, encoding='utf-8')
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i][0:4] == '</S>':
                    line = lines[i - 1]
                    if line[0] == '<':
                        continue
                    s, t = parseTagged(line, fp, i)
                    test_segmented.append(s)
                    test_tag.append(t)
                    test_raw.append(''.join(s))

        elif fp < r"chtb_1152":
            f = open(root + '/' + fp, encoding='utf-8')
            lines = f.readlines()

            for i in range(len(lines)):
                if lines[i][0:4] == '</S>':
                    line = lines[i - 1]
                    if line[0] == '<':
                        continue
                    s, t = parseTagged(line, fp, i)
                    train_segmented.append(s)
                    train_tag.append(t)
                    train_raw.append(''.join(s))
                    # print('train:',len(train_segmented))
                    # print('train:',len(train_tag))
                    # print('train:',len(train_raw))
    train=[train_segmented,train_tag,train_raw]
    dev=[dev_segmented,dev_tag,dev_raw]
    test=[test_segmented,test_tag,test_raw]
    train_pair = []
    for i in range(len(train_segmented)):
        train_pair.append([train_segmented[i],train_tag[i],train_raw[i]])

    # random.shuffle(train_pair)

    train_segmented.clear()
    train_tag.clear()
    train_raw.clear()

    for i in range(len(train_pair)):
        train_segmented.append(train_pair[i][0])
        train_tag.append(train_pair[i][1])
        train_raw.append(train_pair[i][2])

    train = [train_segmented, train_tag, train_raw]

    # print(len(dev_segmented))
    # print(len(test_segmented))
    # print(dev_raw)
    print('load CTB3 successfully')
    return train,dev,test

def loadPeopleDailyData(fp):
    f = open(fp,'r',encoding='utf-8')
    lines = f.readlines()
    data_pair = []
    words = []
    tags = []
    sentences = []
    word_num = 0
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        if len(line)<26:        #two if to get rid of blank line
            continue

        w = []
        t = []
        line_split = line.split('  ')
        for pair_concated in line_split[1:]:
            pair = pair_concated.split('/')
            if pair[0][0] == '[':
                pair[0] = pair[0][1:]

            if ']' in pair[1]:
                pair[1] = pair[1][:pair[1].index(']')]

            if pair[1] == 'na':
                print(pair)
            w.append(pair[0])
            word_num+=1

            # if pair[1] in {'nr','nt','nx','ns'}:
            #     print(line)
            t.append(pair[1].lower())

        words.append(w)
        tags.append(t)
        sentences.append(''.join(w))
    # return words,tags,sentences
    # return words,tags,sentences

    pairs = []
    for i in range(len(words)):
        pairs.append([words[i],tags[i],sentences[i]])

    # random.shuffle(pairs)

    train_pair = pairs[:1500]

    train = []
    train_word = []
    train_tag = []
    train_sentence = []
    for pair in train_pair:
        train_word.append(pair[0])
        train_tag.append(pair[1])
        train_sentence.append(pair[2])


    train = [train_word,train_tag,train_sentence]

    return train

def loadNovelData(fp):
    split = []
    tags = []
    sentences = []
    f = open(fp,encoding='utf-8')
    lines = f.readlines()
    for i in range(len(lines)):
        if lines[i] =='_\n':
            print(fp)
        s,t = parseTagged(lines[i],fp,i)
        split.append(s)
        # t = t.lower()
        tags.append(t)
        sentences.append(''.join(s))

    return split,tags,sentences

def loadAllNovelData(root):
    fps = os.listdir(root)
    split = []
    tags = []
    sentences = []
    for fp in fps:
        f = open(root+'/'+fp,encoding='utf-8')
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i] =='_\n':
                print(fp)
            s,t = parseTagged(lines[i],fp,i)
            split.append(s)
            # t = t.lower()
            tags.append(t)
            sentences.append(''.join(s))

    return split,tags,sentences

def loadNewPeopleDailyData(fp):
    f = open(fp,'r',encoding='utf-8')
    lines = f.readlines()
    segmenteds = []
    tags = []
    sentences = []
    for line in lines:
        line = line.strip()
        segmented = []
        tag = []
        pairs = line.split('\t')
        for pair in pairs:
            s,t = pair.split('/',1)
            if s[0]=='[':
                s = s[1:]
            if ']' in t:
                t = t.replace(']','')
            if '/' in t:
                t = t.split('/')[0]
            segmented.append(s)
            t = t.lower()
            tag.append(t)

        segmenteds.append(segmented)
        tags.append(tag)
        sentences.append(''.join(segmented))

    all = []
    for i in range(len(segmenteds)):
        all.append([segmenteds[i],tags[i],sentences[i]])

    random.seed(1208)
    random.shuffle(all)

    train_pair = all[:len(all)-1500]
    test_pair = all[len(all)-1500:]

    train_segmenteds = []
    train_tags = []
    train_sentences = []

    test_segmenteds = []
    test_tags = []
    test_sentences = []

    for i in range(len(train_pair)):
        train_segmenteds.append(train_pair[i][0])
        train_tags.append(train_pair[i][1])
        train_sentences.append(train_pair[i][2])

    for i in range(len(test_pair)):
        test_segmenteds.append(test_pair[i][0])
        test_tags.append(test_pair[i][1])
        test_sentences.append(test_pair[i][2])


    train = [train_segmenteds,train_tags,train_sentences]
    test = [test_segmenteds,test_tags,test_sentences]

    print('load pku data successfully!')
    return train,test

def loadQiuPKU(fp,seed):
    f = open(fp,'r',encoding='utf-8')
    lines = f.readlines()
    segmenteds = []
    tags = []
    sentences = []
    for line in lines:
        s,t = parseTagged(line)
        segmenteds.append(s)
        tags.append(t)
        sentences.append(''.join(s))


    all = []
    for i in range(len(segmenteds)):
        all.append([segmenteds[i],tags[i],sentences[i]])

    if seed>=0:
        random.seed(seed)
        random.shuffle(all)

    train_pair = all[:len(all)-1500]
    test_pair = all[len(all)-1500:]

    train_segmenteds = []
    train_tags = []
    train_sentences = []

    test_segmenteds = []
    test_tags = []
    test_sentences = []

    for i in range(len(train_pair)):
        train_segmenteds.append(train_pair[i][0])
        train_tags.append(train_pair[i][1])
        train_sentences.append(train_pair[i][2])

    for i in range(len(test_pair)):
        test_segmenteds.append(test_pair[i][0])
        test_tags.append(test_pair[i][1])
        test_sentences.append(test_pair[i][2])


    train = [train_segmenteds,train_tags,train_sentences]
    test = [test_segmenteds,test_tags,test_sentences]

    print('load pku data successfully!')
    return train,test


if __name__ == '__main__':
    new_root = r'D:\wias_nlp\data\qiu\199801.segged_known.txt'
    train,test = loadQiuPKU(new_root)
    for i,s in enumerate(train[0][:10]):
        for j,w in enumerate(s):
            print(w,train[1][i][j])
        print("*************")
    # train,test = loadNewPeopleDailyData(new_root)
    # print(sum(list(map(len,train[2])))/len(train[2]))
    # train = loadPeopleDailyData(r'D:\wias_nlp\data\199801\199802.txt')
    # print(len(train[0]))
    # t = BaseTagger()
    # t.prepareKnowledge(train)
    # print(t.tag_set)
    # print(len(t.tag_set))
    # standard_tag_set = {'ag','a','ad','an',
    #                     'bg','b','dg','c','dg','d','e','f','h','i','j','k','l','mg','m','mg','ng','n','nr','ns',
    #                     'nt','nx','nz','o','p','q','rg','r','s','tg','t','u','vg','v','vd','vn','w','x','yg','y','z'}
    # print(t.tag_set-standard_tag_set)
    # print(standard_tag_set-t.tag_set)
    # exit(1)
    # root = r'D:\yue zhang\ctb8.0\data\postagged'
    # train_n = loadNovelData(r'D:\wias_nlp\data\gold')
    # print(train_n[0])
    # t = BaseTagger()
    # t.prepareKnowledge(train_n)
    # print(t.tag_set)
    # print(len(t.tag_set))
    #
    # t1 = BaseTagger()
    # train_p = loadPeopleDailyData(r'D:\wias_nlp\data\199801\199801.txt')
    # t1.prepareKnowledge(train_p)
    # print(t1.tag_set-t.tag_set)
    # print(t.tag_set-t1.tag_set)

    # standard_tag_set = {'ag','a','ad','an',
    #                     'bg','b','dg','c','dg','d','e','f','h','i','j','k','l','mg','m','mg','ng','n','nr','ns',
    #                     'nt','nx','nz','o','p','q','rg','r','s','tg','t','u','vg','v','vd','vn','w','x','yg','y','z'}
    # people_fp1 =  r'D:\wias_nlp\data\xiaobo\199801.txt'
    # train1 = loadPeopleDailyData(people_fp1)
    # t1 = BaseTagger()
    # t1.prepareKnowledge(train1)
    # print(t1.tag_set)
    # print(len(t1.tag_set))
    #
    # novel_fp1 = r''
    #
    # print(standard_tag_set-t1.tag_set)
    #
    #
    # train_n = loadNovelData(r'D:\wias_nlp\data\gold')
    # # print(train_n[0])
    # t = BaseTagger()
    # t.prepareKnowledge(train_n)
    # print(t.tag_set)
    # print(len(t.tag_set))
    #
    # print(t.tag_set-t1.tag_set)
    # print(t1.tag_set-t.tag_set)