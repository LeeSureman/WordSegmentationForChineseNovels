from feature_lib import *
from logistic_regression import *
from data_preprocess import *
from base_tagger import State,FeatureWeight
import argparse


class WordClassifier(object):
    def __init__(self,noun_tag):
        self.noun2pattern = dict()
        self.noun2pattern_num = dict()
        self.pattern2num = dict()
        self.word_freq = {}
        self.word_freq_sum = 0
        # self.noun_tag = {'n', 'nr', 'ns', 'nz','nt'}
        self.kind_num = dict()
        self.noun_tag = noun_tag
        # self.lr = LR(13, 1)
        for key in self.noun_tag:
            self.kind_num[key] = dict()
        self.lr = LR(13, 1)
        self.positive2tags = dict()

    def prepareKnowledge(self, states):
        for i in range(len(states)):
            for j in range(2, len(states[i].word) - 1):
                if states[i].tag[j] in self.noun_tag:
                    now_pattern = states[i].word[j - 1] + '_' + states[i].word[j + 1]

                    now_pattern_num = self.pattern2num.get(now_pattern)

                    if now_pattern_num:
                        self.pattern2num[now_pattern] += 1
                    else:
                        self.pattern2num[now_pattern] = 1

                    local_pattern = self.noun2pattern.get(states[i].word[j])
                    if local_pattern:
                        local_pattern.add(now_pattern)
                    else:
                        self.noun2pattern[states[i].word[j]] = {now_pattern}

                    local_pattern_num = self.noun2pattern_num.get(states[i].word[j])

                    if local_pattern_num:
                        self.noun2pattern_num[states[i].word[j]] += 1
                    else:
                        self.noun2pattern_num[states[i].word[j]] = 1

                    self.word_freq_sum += 1
                    w_f = self.word_freq.get(states[i].word[j])
                    if w_f:
                        self.word_freq[states[i].word[j]] += 1
                    else:
                        self.word_freq[states[i].word[j]] = 1

                    num_dict = self.kind_num[states[i].tag[j]]
                    num_in_kind = num_dict.get(states[i].word[j])

                    if num_in_kind:
                        num_dict[states[i].word[j]] += 1
                    else:
                        num_dict[states[i].word[j]] = 1

        print('prepare knowledge finish!')

    def prepareData(self, states):
        self.positive_candidate = set()
        self.x = []
        self.y = []
        for tag_kind in self.noun_tag:
            w2n = self.kind_num[tag_kind]
            l = list(w2n.items())
            l.sort(key=lambda x: x[1], reverse=True)
            print(l[0:10])
            l = l[:int(0.3 * len(l))]
            l = list(map(lambda x: x[0], l))
            # for w in l:
            #     self.positive2tags[w] = self.positive2tags.setdefault(w,set()).union({tag_kind})
            self.positive_candidate = self.positive_candidate.union(set(l))
        print('positive noun:', len(self.positive_candidate))
        positive_num = 0
        negative_num = 0
        hasAdd = set()
        for i in range(len(states)):
            for j in range(2, len(states[i].word) - 1):
                w = states[i].word[j]

                if states[i].word[j] in self.positive_candidate and states[i].tag[j] in self.noun_tag:
                    # print('1')
                    if states[i].word[j] not in hasAdd:
                        self.x.append([start_end_punctuation(states[i], j),
                                       count_p_bigger_20(states[i], j, self.noun2pattern),
                                       count_p_between_10_20(states[i], j, self.noun2pattern),
                                       count_p_between_2_10(states[i], j, self.noun2pattern),
                                       count_p_equal_1(states[i], j, self.noun2pattern),
                                       freq_p_bigger_50(states[i], j, self.noun2pattern_num),
                                       freq_p_between_20_50(states[i], j, self.noun2pattern_num),
                                       freq_p_between_5_20(states[i], j, self.noun2pattern_num),
                                       freq_p_smaller_5(states[i], j, self.noun2pattern_num),
                                       PMI_1(states[i].word[j], self.word_freq, self.word_freq_sum),
                                       PMI_n_1(states[i].word[j], self.word_freq, self.word_freq_sum),
                                       PMI_2(states[i].word[j], self.word_freq, self.word_freq_sum),
                                       PMI_n_2(states[i].word[j], self.word_freq, self.word_freq_sum)]
                                      )
                        self.y.append([1])
                        positive_num += 1
                        hasAdd.add(states[i].word[j])
                    negative_tmp = states[i].word[j] + states[i].word[j + 1]
                    if negative_tmp not in self.positive_candidate and negative_tmp not in hasAdd:
                        self.x.append([start_end_punctuation(states[i], j, True),
                                       count_p_bigger_20(states[i], j, self.noun2pattern, True),
                                       count_p_between_10_20(states[i], j, self.noun2pattern, True),
                                       count_p_between_2_10(states[i], j, self.noun2pattern, True),
                                       count_p_equal_1(states[i], j, self.noun2pattern, True),
                                       freq_p_bigger_50(states[i], j, self.noun2pattern_num, True),
                                       freq_p_between_20_50(states[i], j, self.noun2pattern_num, True),
                                       freq_p_between_5_20(states[i], j, self.noun2pattern_num, True),
                                       freq_p_smaller_5(states[i], j, self.noun2pattern_num, True),
                                       PMI_1(negative_tmp, self.word_freq, self.word_freq_sum),
                                       PMI_n_1(negative_tmp, self.word_freq, self.word_freq_sum),
                                       PMI_2(negative_tmp, self.word_freq, self.word_freq_sum),
                                       PMI_n_2(negative_tmp, self.word_freq, self.word_freq_sum)]
                                      )
                        self.y.append([0])
                        negative_num += 1
                        hasAdd.add(negative_tmp)

        print('prepare data finish!')
        print('positive:', positive_num)
        print('negative:', negative_num)
        self.x = np.array(self.x)
        self.y = np.array(self.y)

        for i in range(self.x.shape[0]):
            print('x:',self.x[i],'y:',self.y[i])

        pair = np.concatenate([self.x, self.y], axis=1)
        np.random.seed(100)
        np.random.shuffle(pair)
        train_pair = pair[:int(pair.shape[0] * 0.98)]
        test_pair = pair[int(pair.shape[0] * 0.98):]
        print('train_pair:', train_pair.shape)
        self.train_x = train_pair[:, :13]
        self.train_y = train_pair[:, 13:]
        self.test_x = test_pair[:, :13]
        self.test_y = test_pair[:, 13:]

    def train(self):
        self.lr.fit(self.train_x, self.train_y, epoch=400, minibatch=4000, lr=0.01, test_x=self.test_x, test_y=self.test_y)

    def loadWeight(self, weight_dict):
        self.lr.w = weight_dict['w']
        self.lr.b = weight_dict['b']

    def tag(self, state, j):
        x_list = [start_end_punctuation(state, j),
                  count_p_bigger_20(state, j, self.noun2pattern),
                  count_p_between_10_20(state, j, self.noun2pattern),
                  count_p_between_2_10(state, j, self.noun2pattern),
                  count_p_equal_1(state, j, self.noun2pattern),
                  freq_p_bigger_50(state, j, self.noun2pattern_num),
                  freq_p_between_20_50(state, j, self.noun2pattern_num),
                  freq_p_between_5_20(state, j, self.noun2pattern_num),
                  freq_p_smaller_5(state, j, self.noun2pattern_num),
                  PMI_1(state.word[j], self.word_freq, self.word_freq_sum),
                  PMI_n_1(state.word[j], self.word_freq, self.word_freq_sum),
                  PMI_2(state.word[j], self.word_freq, self.word_freq_sum),
                  PMI_n_2(state.word[j], self.word_freq, self.word_freq_sum)]

        x = np.array(x_list)
        p = self.lr.forward(x)
        return p

    def tag_test(self,left,middle,right):

        w = [left,middle,right]
        t = ['','','']
        state = State(w,t,True)
        j = 3
        x_list = [start_end_punctuation(state, j),
                  count_p_bigger_20(state, j, self.noun2pattern),
                  count_p_between_10_20(state, j, self.noun2pattern),
                  count_p_between_2_10(state, j, self.noun2pattern),
                  count_p_equal_1(state, j, self.noun2pattern),
                  freq_p_bigger_50(state, j, self.noun2pattern_num),
                  freq_p_between_20_50(state, j, self.noun2pattern_num),
                  freq_p_between_5_20(state, j, self.noun2pattern_num),
                  freq_p_smaller_5(state, j, self.noun2pattern_num),
                  PMI_1(state.word[j], self.word_freq, self.word_freq_sum),
                  PMI_n_1(state.word[j], self.word_freq, self.word_freq_sum),
                  PMI_2(state.word[j], self.word_freq, self.word_freq_sum),
                  PMI_n_2(state.word[j], self.word_freq, self.word_freq_sum)]

        x = np.array(x_list)
        p = self.lr.forward(x)
        if p[:,0]>0.5:
            return True
        else:
            return False

    def get_P(self,P):
        self.P = P



class POSClassifier(object):
    def __init__(self,positive_candidate,positive2tags,noun_tag):
        self.positive_candidate = positive_candidate
        self.positive2tags = positive2tags
        self.example_num = 0
        self.tag_dict = {'n':0,'ns':1,'nr':2,'nz':3}
        self.weight = FeatureWeight()
        self.lr = DiscreteLR(self.weight)
        self.noun_tag = noun_tag
    def prepareData(self,states):
        self.id_dict = dict()
        index = 1
        x = []
        y = []
        pair = []
        hasAdd = set()
        for i in range(len(states)):
            for j in range(2,len(states[i].word)-2):
                if states[i].word[j] in self.positive_candidate \
                        and states[i].tag[j] in self.tag_dict:
                    l_1_w = '1'+states[i].word[j-1]
                    l_2_w = '2'+states[i].word[j-2]
                    r_1_w = '3'+states[i].word[j+1]
                    r_2_w = '4'+states[i].word[j+2]
                    l_1_t = '5'+states[i].tag[j-1]
                    r_1_t = '6'+states[i].tag[j+1]
                    first_c = '7'+states[i].word[j][0]
                    last_c = '8'+states[i].word[j][-1]
                    if l_1_w+l_2_w+r_1_w+r_2_w not in hasAdd:
                        pair.append([l_1_w,l_2_w,r_1_w,r_2_w,l_1_t,r_1_t,first_c,last_c,states[i].tag[j]])
                        hasAdd.add(l_1_w+l_2_w+r_1_w+r_2_w)
                        self.example_num += 1

        random.seed(20008)
        random.shuffle(pair)

        self.train_set = pair[:len(pair)-5000]
        self.test_set = pair[len(pair)-5000:]

        print('example_num:',len(hasAdd))

        print('pos data prepare finish!')

    def train_lr(self):
        y = []
        for i in range(len(self.train_set)):
            try:
                y.append(self.tag_dict[self.train_set[i][-1]])
            except KeyError as e:
                print(self.train_set[i])

        y = np.array(y)
        y = np.eye(4)[y]

        test_y = []
        for i in range(len(self.test_set)):
            test_y.append(self.tag_dict[self.test_set[i][-1]])

        test_y = np.array(test_y)
        test_y = np.eye(4)[test_y]


        self.lr.fit(self.train_set,y,101,1000,0.01,self.test_set,test_y)

    def train(self,epoch):
        self.weight = FeatureWeight()
        n_error = 0
        for e in range(epoch):
            for i in range(len(self.train_set)):
                max_t = 'n'
                max_score = 0
                now_score = 0
                for t in self.noun_tag:
                    for j in range(8):
                        now_score+=self.weight.getFeatureScore(t+self.train_set[i][j],isTrain=True)

                    # now_score+=self.weight.getFeatureScore(t+'b',isTrain=True)
                    now_score+=self.weight.getFeatureScore(t+'b',isTrain=True)
                    if now_score>max_score:
                        max_score = now_score
                        max_t = t
                    now_score = 0
                # print(i)
                if max_t!=self.train_set[i][-1]:
                    n_error+=1
                    for j in range(8):
                        self.weight.updateFeatureScore(max_t+self.train_set[i][j],-1,e)
                        self.weight.updateFeatureScore(self.train_set[i][-1]+self.train_set[i][j],1,e)

                    self.weight.updateFeatureScore(max_t+'b',-1,e)
                    self.weight.updateFeatureScore(self.train_set[i][-1] + 'b', 1, e)
            print('error:',n_error,'/',len(self.train_set))
            if n_error==0:
                print('n_error=0,early quit!')
                break
            n_error = 0
            # self.weight.useRaw()
            if e %10==0:
                self.weight.useRaw()
                self.test()
                self.weight.accumulateAll(e+1)
                self.weight.useAverage(e+1)
                self.test()
        print('train finish !')

    def test(self,test_set=None):
        if not test_set:
            test_set = self.test_set

        correct_num = 0
        for i in range(len(test_set)):
            max_score = 0
            max_t = 'n'
            now_score = 0
            for t in self.noun_tag:
                for j in range(8):
                    now_score+=self.weight.getFeatureScore(t+test_set[i][j],False)

                if now_score>max_score:
                    max_t = t
                    max_score = now_score
                now_score = 0

            if max_t == test_set[i][-1]:
                correct_num+=1


        print('\nacc:',correct_num/len(test_set),'\n')










if __name__ == '__main__':
    wc = WordClassifier()
    pku_root = r'D:\wias_nlp\data\pku98\199801simplified.txt'
    train, test = loadNewPeopleDailyData(pku_root)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', help='train or test ')
    # parser.add_argument('--weight', help='where to place weight')
    # args = parser.parse_args()


    states = []
    for i in range(len(train[0])):
        states.append(State(train[0][i],train[1][i],True))
    wc.prepareKnowledge(states)
    wc.prepareData(states)
    # pc = POSClassifier(wc.positive_candidate,wc.positive2tags)
    # pc.prepareData(states)
    # pc.train_lr()
    # pc.prepareData(states)
    # pc.train(500)
# pc.test()


# wc.prepareKnowledge(states)
# wc.prepareData(states)
#
# print(wc.train_x.shape)
# print(wc.train_y.shape)
# print(wc.test_x.shape)
# print(wc.test_y.shape)


# print(len(wc.x))
    wc.lr.fit(wc.train_x,wc.train_y,epoch=100,minibatch=4000,lr=0.001,test_x=wc.test_x,test_y=wc.test_y)
    while True:
        line = input()
        a,b,c = line.split()
        print(wc.tag_test(a,b,c))