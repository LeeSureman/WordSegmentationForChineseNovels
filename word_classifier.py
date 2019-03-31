import numpy as np
from data_preprocess import *
import sklearn as sk
import argparse
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score
from data_preprocess import *
import copy
import math
from enhanced_tagger_1 import EnhancedTagger

np.set_printoptions(suppress=True, threshold=np.nan)



def get_my_pmi(w,pair_freq,info):
    info.my_pmi = 0
    if len(w) > 1:
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                info.my_pmi += pair_freq.setdefault(w[i] + w[j], 0)

        info.my_pmi /= (len(w) * (len(w) - 1))

    return info.my_pmi

def get_char_freq(w,char_freq,info):
    for c in w:
        info.char_freq_average += char_freq.setdefault(c, 0)
    info.char_freq_average /= len(w)
    return info.char_freq_average

class Info:
    def __init__(self):
        self.pattern_set = set()
        self.pattern_list = []
        self.starts_punc = 0
        self.ends_punc = 0
        self.starts_ends_punc = 0
        self.my_pmi = 0
        self.char_freq_average = 0

class WordClassifier:
    def __init__(self, train_fp,limit=0.5,standarized=False,use_my_feature=False,use_start_end_pmi=False,
                 use_lisan=False,use_punc=False):
        ''' train_fp:the path of the pku training set
            limit: the threshold of the substring to be reffered as a new noun
            standarized: whether to standarize the x
            use_my_feature:whether to use the feature out of the paper,just set it False
            use_start_end_pmi:the pmi of the start and end of the word, out of the paper,too.
                                but is useful
            use_lisan:whether to discretize the pattern count and freq.
                    how ever, not like in the paper, my experiment shows not discretizing them is better

            use_punc:whether to use them makes no difference


        '''
        punctuations = '！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕'
        self.gen_set = set()
        self.punctuation = set()
        for p in punctuations:
            self.punctuation.add(p)
        self.use_punc = use_punc
        self.use_lisan = use_lisan
        self.use_start_end_pmi = use_start_end_pmi
        self.total_num = 0
        self.use_my_feature = use_my_feature
        self.standarized=standarized
        self.limit = limit
        self.l = 15
        self.train_fp = train_fp
        self.noun_tags = {'n', 'nr', 'nz', 'nt', 'ns'}
        self.positive_x = {}
        self.negative_x = {}
        self.word2freq_dict = {}
        self.word2freq_list = {}
        self.char_freq = {}             #the freq of a char occuring in a word
        self.pair_freq = {}             #the freq of two chars(even unconsecutive) occuring in a word,
        self.P = dict()
        self.l_r = LR(tol=0.1,solver='sag',C=1)

        self.s_s = StandardScaler()
        self.w_candidate_info = dict()
        self.W = set()
        self.seg_freq = dict()
        for key in self.noun_tags:
            self.word2freq_dict[key] = {}
            self.word2freq_list[key] = []

    def get_start_end_pmi(self,w):
        if len(w)>1:
            p1 = self.seg_freq.setdefault(w[0],0)
            p2 = self.seg_freq.setdefault(w[-1],0)
            p3 = self.seg_freq.setdefault(w[0]+w[-1],0)

            if p1==0 or p2==0 or p3==0:
                pmi = 0
            else:
                pmi = math.log((p3*self.total_num)/(p1*p2))
            return pmi
        else:
            return 0

    def prepare_training_data_and_train(self):
        ''' prepare training data and
            train

        '''
        segmenteds, tags, sentences = loadQiuPKU(self.train_fp, -2)
        # self.wordss = segmenteds
        # self.tagss = tags
        # self.sentences = sentences
        for i, words in enumerate(segmenteds):
            for j, w in enumerate(words):
                self.gen_set.add(w)
                if tags[i][j] in self.noun_tags:
                    if w in self.word2freq_dict[tags[i][j]]:
                        self.word2freq_dict[tags[i][j]][w] += 1
                    else:
                        self.word2freq_dict[tags[i][j]][w] = 1

                    if len(w)>1:
                        if self.seg_freq.get(w[0]) is None:
                            self.seg_freq[w[0]] = 1
                        else:
                            self.seg_freq[w[0]]+=1

                        if self.seg_freq.get(w[-1]) is None:
                            self.seg_freq[w[-1]] = 1
                        else:
                            self.seg_freq[w[-1]]+=1
                        if self.seg_freq.get(w[0]+w[-1]) is None:
                            self.seg_freq[w[0]+w[-1]] = 1
                        else:
                            self.seg_freq[w[0]+w[-1]]+=1
                        self.total_num+=1

        for tag in self.word2freq_dict:
            for word, freq in self.word2freq_dict[tag].items():
                self.word2freq_list[tag].append([word, freq])

        for tag in self.noun_tags:
            self.word2freq_list[tag].sort(key=lambda a: a[1],reverse=True)
            print('tag:',tag)
            for pair in self.word2freq_list[tag][:int(0.3 * len(self.word2freq_list[tag]))]:
                print(pair)
                self.positive_x[pair[0]] = Info()




        for i, words in enumerate(segmenteds):
            for j, w in enumerate(words):
                if w in self.positive_x:
                    tmp_info = self.positive_x[w]
                    tmp_pattern = (" " if j == 0 else words[j - 1]) + '-' +\
                                  (" " if j == len(words) - 1 else words[j + 1])
                    if self.P.get(tmp_pattern) is None:
                        self.P[tmp_pattern] = 1
                    else:
                        self.P[tmp_pattern] += 1
                    tmp_info.pattern_set.add(tmp_pattern)
                    tmp_info.pattern_list.append(tmp_pattern)
                    s_w = False
                    e_w = False
                    if j > 0:
                        if tags[i][j - 1] == 'w':
                            s_w = True
                            tmp_info.starts_punc += 1

                    if j < len(tags[i]) - 1:
                        if tags[i][j + 1] == 'w':
                            e_w = True
                            tmp_info.ends_punc += 1

                    if e_w and s_w:
                        tmp_info.starts_ends_punc += 1

        #prepare the negative examples
        for i, words in enumerate(segmenteds):
            for j, w in enumerate(words):
                if w in self.positive_x:
                    if (j > 0 and words[j - 1] + w not in self.positive_x):
                        self.negative_x[words[j - 1] + w] = Info()

                    if j < len(words) - 1 and w + words[j + 1] not in self.positive_x:
                        self.negative_x[w + words[j+1]] = Info()

        for i, words in enumerate(segmenteds):
            for j, w in enumerate(words):
                if j < len(words) - 1:
                    tmp_negative = words[j] + words[j + 1]
                    if tmp_negative in self.negative_x:
                        tmp_info = self.negative_x[tmp_negative]

                    else:
                        continue

                    if j == 0:
                        if len(words)<3:
                            if ' '+'-'+' ' in self.P:
                                tmp_info.pattern_set.add(' '+'-'+' ')
                                tmp_info.pattern_list.append(' '+'-'+' ')

                            tmp_info.starts_punc += 1
                            continue
                        if ' ' + '-' + words[2] in self.P:
                            tmp_info.pattern_set.add(' ' + '-' + words[2])
                            tmp_info.pattern_list.append(' ' + '-' + words[2])
                        tmp_info.starts_punc += 1
                        if tags[i][j + 2] == 'w':
                            tmp_info.ends_punc += 1
                            tmp_info.starts_ends_punc += 1

                    elif j == len(words) - 2:
                        if words[j - 1] + '-'+' ' in self.P:
                            tmp_info.pattern_set.add(words[j - 1] + '-'+' ')
                            tmp_info.pattern_list.append(words[j - 1] + '-'+' ')

                        tmp_info.ends_punc += 1
                        if tags[i][j - 1] == 'w':
                            tmp_info.starts_punc += 1
                            tmp_info.starts_ends_punc += 1
                    else:
                        if words[j-1] + '-'+words[j + 2] in self.P:
                            tmp_info.pattern_set.add(words[j-1] + '-'+words[j + 2])
                            tmp_info.pattern_list.append(words[j-1] + '-'+words[j + 2])

                        e_w = False
                        s_w = False
                        if tags[i][j - 1] == 'w':
                            tmp_info.starts_punc += 1
                            s_w = True

                        if tags[i][j + 2] == 'w':
                            tmp_info.ends_punc += 1
                            e_w = True

                        if e_w and s_w:
                            tmp_info.starts_ends_punc += 1

        for w in self.positive_x:
            for i in range(len(w)):
                for j in range(i+1,len(w)):
                    if self.pair_freq.get(w[i]+w[j]) is None:
                        self.pair_freq[w[i]+w[j]] = 1
                    else:
                        self.pair_freq[w[i] + w[j]] += 1

        #the following is to make pair_freq symmetric

        new_pair_freq = dict()
        hasAdd = set()
        for w,freq in self.pair_freq.items():
            if w in hasAdd:
                print(w)
                continue
            if w[1]+w[0] in self.pair_freq:
                tmp_freq = self.pair_freq[w[0]+w[1]]+self.pair_freq[w[1]+w[0]]
            else:
                tmp_freq = self.pair_freq[w[0]+w[1]]
            new_pair_freq[w[0]+w[1]] = tmp_freq
            new_pair_freq[w[1]+w[0]] = tmp_freq
            hasAdd.add(w[0]+w[1])
            hasAdd.add(w[1]+w[0])


        print('old_pair_freq:',len(self.pair_freq))
        self.pair_freq = new_pair_freq
        print('new_pair_freq:',len(self.pair_freq))

        for w in self.positive_x:
            for c in w:
                if c in self.char_freq:
                    self.char_freq[c] += 1
                else:
                    self.char_freq[c] = 1

        N_positive = len(self.positive_x)
        N_negative = len(self.negative_x)
        x_np_positive = np.zeros(shape=[N_positive,self.l])
        y_np_positive = np.zeros(shape=[N_positive,1])
        x_np_negative = np.zeros(shape=[N_negative,self.l])
        y_np_negative = np.zeros(shape=[N_negative,1])

        for i,(w,info) in enumerate(self.positive_x.items()):
            y_np_positive[i][0] = 1
            if self.use_lisan:
                if len(info.pattern_set)==1:
                    x_np_positive[i][0] = 1
                elif len(info.pattern_set)<10:
                    x_np_positive[i][1] = 1
                elif len(info.pattern_set)<20:
                    x_np_positive[i][2] = 1
                elif len(info.pattern_set)>=20:
                    x_np_positive[i][3] = 1

                if len(info.pattern_list)<5:
                    x_np_positive[i][4] = 1
                elif len(info.pattern_list)<20:
                    x_np_positive[i][5] = 1
                elif len(info.pattern_list)<50:
                    x_np_positive[i][6] = 1
                elif len(info.pattern_list)>=50:
                    x_np_positive[i][7] = 1
            else:
                x_np_positive[i][0] = len(info.pattern_set)
                x_np_positive[i][1] = len(info.pattern_list)


            if self.use_my_feature:
                x_np_positive[i][8] = get_char_freq(w, self.char_freq, info)
                x_np_positive[i][9] = get_my_pmi(w,self.pair_freq,info)
            if self.use_start_end_pmi:
                x_np_positive[i][10] = self.get_start_end_pmi(w)
            if self.use_punc:
                x_np_positive[i][11] = int(info.starts_ends_punc>=2)


        for i, (w, info) in enumerate(self.negative_x.items()):
            y_np_negative[i][0] = 0

            if self.use_lisan:
                if len(info.pattern_set) == 1:
                    x_np_negative[i][0] = 1
                elif len(info.pattern_set) < 10:
                    x_np_negative[i][1] = 1
                elif len(info.pattern_set) < 20:
                    x_np_negative[i][2] = 1
                elif len(info.pattern_set) >= 20:
                    x_np_negative[i][3] = 1

                if len(info.pattern_list) < 5:
                    x_np_negative[i][4] = 1
                elif len(info.pattern_list) < 20:
                    x_np_negative[i][5] = 1
                elif len(info.pattern_list) < 50:
                    x_np_negative[i][6] = 1
                elif len(info.pattern_list) >= 50:
                    x_np_negative[i][7] = 1
            else:
                x_np_negative[i][0] = len(info.pattern_set)
                x_np_negative[i][1] = len(info.pattern_list)


            if self.use_my_feature:
                x_np_negative[i][8] = get_char_freq(w,self.char_freq,info)
                x_np_negative[i][9] = get_my_pmi(w,self.pair_freq,info)

            if self.use_start_end_pmi:
                x_np_negative[i][10] = self.get_start_end_pmi(w)

            if self.use_punc:
                x_np_negative[i][11] = int(info.starts_ends_punc>=2)
        # for w, info in self.positive_x.items():
        #     print(w, '\t', 'count:', len(info.pattern_set), 'freq:', len(info.pattern_list),
        #           'char_freq:', info.char_freq_average, 'my_pmi:', info.my_pmi, '\t', info.pattern_set, file=f)
        #
        # for w, info in self.negative_x.items():
        #     print(w, '\t', 'count:', len(info.pattern_set), 'freq:', len(info.pattern_list),
        #           'char_freq:', info.char_freq_average, 'my_pmi:', info.my_pmi, '\t', info.pattern_set, file=f)

        # self.x_np = np.concatenate([x_np_positive,x_np_negative],axis=0)
        # self.y_np = np.concatenate([y_np_positive,y_np_negative],axis=0)
        #
        self.x_np = np.concatenate([np.tile(x_np_positive,[25,1]),x_np_negative],axis=0)
        self.y_np = np.concatenate([np.tile(y_np_positive,[25,1]),y_np_negative],axis=0)
        #
        # self.x_np = np.concatenate([x_np_positive,x_np_negative[:x_np_positive.shape[0]]],axis=0)
        # self.y_np = np.concatenate([y_np_positive,y_np_negative[:x_np_positive.shape[0]]],axis=0)

        self.x_np_train,self.x_np_test,self.y_np_train,self.y_np_test = \
        train_test_split(self.x_np,self.y_np,test_size=0.01,random_state=1208)
        if self.standarized:
            self.x_np_train = self.s_s.fit_transform(self.x_np_train)
            self.x_np_test = self.s_s.transform(self.x_np_test)
        # self.x_np_train = np.concatenate([np.tile(x_np_positive,[22,1]),self.x_np_train],axis=0)
        # self.y_np_train = np.concatenate([np.tile(y_np_positive,[22,1]),self.y_np_train],axis=0)
        self.l_r.fit(self.x_np_train,self.y_np_train)

        tmp = self.l_r.score(self.x_np_test,self.y_np_test)

        print(self.x_np_train.shape)

        predict = self.l_r.predict(self.x_np_test)

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(predict.shape[0]):
            if predict[i]>0:
                if self.y_np_test[i][0]==1:
                    tp+=1
                else:
                    fp+=1
            else:
                if self.y_np_test[i][0]==1:
                    fn+=1
                else:
                    tn+=1

        print('weight:',self.l_r.coef_)
        print('bias:',self.l_r.intercept_)
        print('tp:',tp,'fp:',fp,'tn:',tn,'fn:',fn)
        p = tp/(tp+fp)
        r = tp/(tp+fn)

        print('p:',tp/(tp+fp))
        print('r:',tp/(tp+fn))
        print('f1:',2*p*r/(p+r))
        print('acc:',(tp+tn)/(tp+fp+tn+fn))


        print(tmp)

    def predict_a_word(self,w):
        ''' input:a word:str
            output:whether it is recoginized as a word
            True or False
        '''
        info = self.w_candidate_info[w]
        x = np.zeros([1,self.l])
        if self.use_lisan:

            if len(info.pattern_set) == 1:
                x[0][0] = 1
            elif len(info.pattern_set) < 10:
                x[0][1] = 1
            elif len(info.pattern_set) < 20:
                x[0][2] = 1
            elif len(info.pattern_set) >= 20:
                x[0][3] = 1

            if len(info.pattern_list) < 5:
                x[0][4] = 1
            elif len(info.pattern_list) < 20:
                x[0][5] = 1
            elif len(info.pattern_list) < 50:
                x[0][6] = 1
            elif len(info.pattern_list) >= 50:
                x[0][7] = 1
        else:
            x[0][0] = len(info.pattern_set)
            x[0][1] = len(info.pattern_list)

        if self.use_my_feature:
            x[0][8] = get_char_freq(w, self.char_freq, info)
            x[0][9] = get_my_pmi(w, self.pair_freq, info)
        if self.use_start_end_pmi:
            x[0][10] = self.get_start_end_pmi(w)

        if self.use_punc:
            x[0][11] = info.starts_ends_punc>=2

        if self.standarized:
            x = self.s_s.transform(x)

        p = self.l_r.predict_proba(x)
        if(p[0][1]>self.limit):
            return True
        else:
            return False


        # print(classification_report(, lr_y_predict, target_names=['Benign', 'Maligant'])

    def get_feature_prob(self,w,info_dict):
        ''' input:a word:str
            output:its features and the prob it is recgnized as a word
        '''

        info = info_dict[w]
        x = np.zeros([1,self.l])
        if self.use_lisan:
            if len(info.pattern_set) == 1:
                x[0][0] = 1
            elif len(info.pattern_set) < 10:
                x[0][1] = 1
            elif len(info.pattern_set) < 20:
                x[0][2] = 1
            elif len(info.pattern_set) >= 20:
                x[0][3] = 1

            if len(info.pattern_list) < 5:
                x[0][4] = 1
            elif len(info.pattern_list) < 20:
                x[0][5] = 1
            elif len(info.pattern_list) < 50:
                x[0][6] = 1
            elif len(info.pattern_list) >= 50:
                x[0][7] = 1
        else:
            x[0][0] = len(info.pattern_set)
            x[0][1] = len(info.pattern_list)


        if self.use_my_feature:
            x[0][8] = get_char_freq(w, self.char_freq, info)
            x[0][9] = get_my_pmi(w, self.pair_freq, info)
        if self.use_start_end_pmi:
            x[0][10] = self.get_start_end_pmi(w)

        if self.use_punc:
            x[0][11] = int(info.starts_ends_punc>=2)


        if self.standarized:
            x = self.s_s.transform(x)
        p = self.l_r.predict_proba(x)

        return p,x


def on_new_pattern(pattern,info_dict,wordss):
    '''the function is called when the new pattern is found'''
    for i, words in enumerate(wordss):
        for j, w in enumerate(words):
            if w in info_dict:
                now_pattern = (" " if j == 0 else words[j - 1]) + '-' + \
                             (" " if j == len(words) - 1 else words[j + 1])
                if now_pattern==pattern:
                    print('找到了',w,'有',now_pattern,'这个pattern')
                    info_dict[w].pattern_set.add(now_pattern)
                    info_dict[w].pattern_list.append(now_pattern)

def get_all_pattern(target,wordss):
    '''return the targets's all pattern'''
    w_patterns = set()
    for i,words in enumerate(wordss):
        for j, w in enumerate(words):
            if w==target:
                now_pattern = (" " if j == 0 else words[j - 1]) + '-' + \
                             (" " if j == len(words) - 1 else words[j + 1])
                w_patterns.add(now_pattern)

    return w_patterns

def has_no_punc(w,puncs=set()):
    for c in w:
        if c in puncs:
            return False
        # if not ((c>='\u4e00' and c<='\u9fff') or (c<='z' and c>='a')):
        #     return False

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',help='the path of the training set')
    parser.add_argument('--auto_tagged',default=None,help='the path of the auto tagged novel')
    # parser.add_argument('--pku_dict',default=r'D:\PycharmProjects\WordSegmentationForChineseNovels\data\qiu_dict.txt')
    parser.add_argument('--gold',help='the path of the gold novel')
    # parser.add_argument('--auto_tagged',default=r'D:\PycharmProjects\WordSegmentationForChineseNovels\data\novel\zx_300_dev.txt')
    parser.add_argument('--data_seed',type=int,default=-1,help='how to shuffle data train/test,if -1 the last 1500 sentences are in test set')

    parser.add_argument('--test', default=None, help='if mode is test,it is annotated,if mode is tag,it is raw')
    parser.add_argument('--base_weight',required=True,default=None,help='base tanggerweight')
    parser.add_argument('--enhanced_weight',required=True,help='enhanced tagger weight')
    parser.add_argument('--dataset_test',help='ctb or pku or novel')
    # parser.add_argument('--mode',help='train or test')
    parser.add_argument('--pku_dict',help='the common word in pku dict',required=True)
    parser.add_argument('--add',default='')
    parser.add_argument('--dataset_train', help='ctb or pku')
    parser.add_argument('--new_feature',default='1')
    # parser.add_argument('--record',default=None)
    parser.add_argument('--use_pattern_feature',default=False,help='whether use pattern feature')
    parser.add_argument('--use_closed_set', default='0', help='whether to use penn closed set tag')

    parser.add_argument('--output_tagged',help='the path of the tagged sentenced',default=None)
    parser.add_argument('--is_raw',default=False,help='the data format when tagging',type=bool)
    parser.add_argument('--mode',help='test or tag')



    args = parser.parse_args()
    w_c = WordClassifier(args.train,use_start_end_pmi=True,use_lisan=False,use_punc=True,
                         limit=0.5)
    w_c.prepare_training_data_and_train()
    train, test = loadQiuPKU(args.train, args.data_seed)
    b_t = BaseTagger(args)
    b_t.prepareKnowledge(train)
    b_t.weight.weightDict = pickle.load(open(args.base_weight, 'rb'))



    # wordss,tagss,sentences = loadNovelData(args.auto_tagged)

    gold_wordss,gold_tagss,gold_sentences = loadNovelData(args.gold)

    states = []
    wordss = []
    tagss = []
    sentences = gold_sentences
    for s in gold_sentences:
        tmp_state = b_t.tag(s,False,b_t.judge_by_rule(s))
        wordss.append(tmp_state.word[2:-1])
        tagss.append(tmp_state.tag[2:-1])
    for words in wordss:
        for w in words:
            if w not in w_c.gen_set and w not in w_c.w_candidate_info:
                w_c.w_candidate_info[w] = Info()
                get_char_freq(w,w_c.char_freq,w_c.w_candidate_info[w])
                get_my_pmi(w,w_c.char_freq,w_c.w_candidate_info[w])

    #the following is constructing novel word features
    for i,words in enumerate(wordss):
        for j,w in enumerate(words):
            if w in w_c.w_candidate_info:
                now_pattern = (" " if j == 0 else words[j - 1]) + '-' + \
                              (" " if j == len(words) - 1 else words[j + 1])

                now_info = w_c.w_candidate_info[w]
                if now_pattern in w_c.P:
                    now_info.pattern_set.add(now_pattern)
                    now_info.pattern_list.append(now_pattern)

                s_w = False
                e_w = False
                if j==0:
                    s_w = True
                    if len(words)<2:
                        e_w = True
                    else:
                        if wordss[i][j+1] in w_c.punctuation:
                            e_w = True
                elif j == len(words)-1:
                    e_w = True
                    if wordss[i][j-1] in w_c.punctuation:
                        s_w = True
                else:
                    if wordss[i][j-1] in w_c.punctuation:
                        s_w = True
                    if wordss[i][j+1] in w_c.punctuation:
                        e_w = True

                if s_w ==True:
                    now_info.starts_punc+=1

                if e_w == True:
                    now_info.ends_punc+=1

                if e_w and s_w:
                    now_info.starts_ends_punc+=1

    while True:
        print('\n'+'*'*20+'\n')
        tmp_candidate_info = w_c.w_candidate_info.copy()
        old_size = len(w_c.w_candidate_info)
        for w in tmp_candidate_info:
            if w_c.predict_a_word(w) and has_no_punc(w,w_c.punctuation):
                del w_c.w_candidate_info[w]
                w_c.W.add(w)
                print(w,'被识别为新的名词',w_c.get_feature_prob(w,tmp_candidate_info))
                print(tmp_candidate_info[w].pattern_set)
                print(tmp_candidate_info[w].pattern_list)

                w_patterns = get_all_pattern(w,wordss)

                for p in w_patterns:
                    if p not in w_c.P:
                        w_c.P[p] = 1
                        print(w,'有',p,'这个pattern，所以：')

                        on_new_pattern(p,tmp_candidate_info,wordss)

        if len(w_c.w_candidate_info) == old_size:
            break



    # print(w_c.W)

    all_new_noun = set()

    for i,words in enumerate(gold_wordss):
        for j,w in enumerate(words):
            if w not in w_c.gen_set and gold_tagss[i][j] in w_c.noun_tags:
            # if w not in w_c.gen_set:
                all_new_noun.add(w)
    # print(all_new_noun)
    true_w = all_new_noun.intersection(w_c.W)

    precision = len(true_w)/len(w_c.W)
    recall = len(true_w)/len(all_new_noun)
    if ' ' in w_c.W:
        w_c.W.remove(' ')
    print('we:',len(w_c.W))
    print('gold:',len(all_new_noun))
    print('r:',recall,'p:',precision)
    print('f1:',2*recall*precision/(recall+precision))

    print('过滤出100个频率最高的词')
    w2freq_dict = dict()
    for i,words in enumerate(wordss):
        for j,w in enumerate(words):
            if w in w_c.W:
                if w2freq_dict.get(w) is None:
                    w2freq_dict[w] = 1
                else:
                    w2freq_dict[w]+=1

    w2freq_list = list(w2freq_dict.items())
    w2freq_list.sort(key=lambda b:b[1],reverse=True)
    w2freq_list = w2freq_list[:100]
    filted_w = set()
    for w,freq in w2freq_list:
        filted_w.add(w)
    true_w = all_new_noun.intersection(filted_w)
    # for w in
    precision = len(true_w)/len(filted_w)
    recall = len(true_w)/len(all_new_noun)
    print('we:',len(filted_w))
    print('gold:',len(all_new_noun))
    print('r:',recall,'p:',precision)
    print('f1:',2*recall*precision/(recall+precision))
    len2freq = {}
    for p in w_c.P:
        c1,c2 = p.split('-')
        len2freq[len(c2)] = len2freq.setdefault(len(c2),0)+1
    print(len2freq)

    w2freq = {}
    for w in w_c.W:
        w2freq[len(w)] = w2freq.setdefault(len(w),0)+1
    print(w2freq)



    e_t = EnhancedTagger(args)

    if args.mode =='test':

        train,test = loadQiuPKU(args.train,args.data_seed)


        novel_test = loadNovelData(args.test)




        # n_count = 0
        # genCorpus_word = set()
        # for i in range(len(train[0])):
        #     for j in range(len(train[0][i])):
        #         genCorpus_word.add(train[0][i][j])

        # auto_split = []
        # W_candidate = set()
        # for i in range(len(novel_test[0])):
        #     now_state = b_t.tag(novel_test[2][i],False,b_t.judge_by_rule(novel_test[2][i]))
        #     auto_split.append(now_state)
        #     for i in range(2,len(now_state.word)-1):
        #         if now_state.word[i] not in genCorpus_word:
        #             W_candidate.add(now_state.word[i]+str(int(now_state.tag[i-1]=='w'and now_state.tag[i+1] =='w')))

        # print(W_candidate)


        e_t.prepareKnowledge(train)
        e_t.weight.weightDict = pickle.load(open(args.enhanced_weight,'rb'))

        e_t.W = w_c.W
        e_t.P = w_c.P

        ground_true_noun = set()
        for i,words in enumerate(gold_wordss):
            for j,w in enumerate(words):
                if gold_tagss[i][j] in w_c.noun_tags:
                    ground_true_noun.add(gold_wordss[i][j])

        e_t.W = ground_true_noun

        novel_gold_state = []
        for i in range(len(novel_test[0])):
            novel_gold_state.append(State(novel_test[0][i], novel_test[1][i], isGold=True))


        b_result = b_t.test(novel_test[2],novel_gold_state)

        e_result = e_t.test(novel_test[2],novel_gold_state)

        print('base tagger:',b_result)
        print('\nenhanced tagger:',e_result)

        # print('w:',e_t.wc.lr.w)
        # print('b:',e_t.wc.lr.b)

    elif args.mode =='tag' :
        # train,test = loadQiuPKU(args.train,args.data_seed)

        if args.is_raw:
            # print('israw!')
            novel_fp = open(args.test, 'r', encoding='utf-8')
            novel_raw = novel_fp.readlines()

        else:
            # print('is not raw!')
            novel_test = loadNovelData(args.test)
        # novel_fp = open(args.test,'r',encoding='utf-8')
            novel_raw = novel_test[2]

        train,test = loadQiuPKU(args.train,args.data_seed)


        # novel_test = loadNovelData(args.test)


        b_t.prepareKnowledge(train)
        b_t.weight.weightDict = pickle.load(open(args.base_weight,'rb'))



        # n_count = 0
        # genCorpus_word = set()
        # for i in range(len(train[0])):
        #     for j in range(len(train[0][i])):
        #         genCorpus_word.add(train[0][i][j])

        # auto_split = []
        # W_candidate = set()
        # for i in range(len(novel_test[0])):
        #     now_state = b_t.tag(novel_test[2][i],False,b_t.judge_by_rule(novel_test[2][i]))
        #     auto_split.append(now_state)
        #     for i in range(2,len(now_state.word)-1):
        #         if now_state.word[i] not in genCorpus_word:
        #             W_candidate.add(now_state.word[i]+str(int(now_state.tag[i-1]=='w'and now_state.tag[i+1] =='w')))

        # print(W_candidate)


        e_t.prepareKnowledge(train)
        e_t.weight.weightDict = pickle.load(open(args.enhanced_weight,'rb'))

        e_t.W = w_c.W
        e_t.P = w_c.P

        ground_true_noun = set()
        for i,words in enumerate(gold_wordss):
            for j,w in enumerate(words):
                if gold_tagss[i][j] in w_c.noun_tags:
                    ground_true_noun.add(gold_wordss[i][j])

        # e_t.W = ground_true_noun

        novel_gold_state = []
        for i in range(len(novel_test[0])):
            novel_gold_state.append(State(novel_test[0][i], novel_test[1][i], isGold=True))


        # b_result = b_t.test(novel_test[2],novel_gold_state)
        #
        # e_result = e_t.test(novel_test[2],novel_gold_state)
        #
        # print('base tagger:',b_result)
        # print('\nenhanced tagger:',e_result)

        if args.output_tagged==None:
            f = open('tmp_result','a',encoding='utf-8')
            print('********new tag result********',file=f)
        else:
            f = open(args.output_tagged,'w',encoding='utf-8')
        for line in novel_raw:
            tmp_r = e_t.tag(line,False,e_t.judge_by_rule(line))
            now_line = []
            for i,w in enumerate(tmp_r.word):
                if tmp_r.tag[i]!='PAD' and w!=' ':
                    now_line.append(w+'_'+tmp_r.tag[i])
            print(' '.join(now_line),file=f)
        # noun_acc = len(novel_noun_set & e_t.W) / len(e_t.W)

        # b_result = b_t.test(novel_test[2],novel_gold_state)

        # e_result = e_t.test(novel_test[2],novel_gold_state)

        # print('noun acc:',noun_acc)


        # print('base tagger:',b_result)
        # print('\nenhanced tagger:',e_result)










