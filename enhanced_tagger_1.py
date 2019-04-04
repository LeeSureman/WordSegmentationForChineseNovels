from data_preprocess import *
from feature_lib import *
from enum import Enum
import copy
import time
import pickle
import argparse
from classifiers import WordClassifier,POSClassifier

def countCorrect(predict,gold):
    predict_word = predict.word[2:-1]
    predict_tag = predict.tag[2:-1]
    gold_word = gold.word[2:-1]
    gold_tag = gold.tag[2:-1]
    predict_all_num = len(predict_word)
    gold_all_num = len(gold_word)
    seg_correct_num = 0
    joint_correct_num = 0
    gold_index = 0
    gold_charindex = 0
    predict_index = 0
    predict_charindex = 0
    while gold_index<gold_all_num and predict_index<predict_all_num:
        if predict_word[predict_index] == gold_word[gold_index]:
            seg_correct_num+=1
            if predict_tag[predict_index] == gold_tag[gold_index]:
                joint_correct_num+=1
            predict_charindex+=len(predict_word[predict_index])
            gold_charindex+=len(gold_word[gold_index])
            predict_index+=1
            gold_index+=1
        else:
            if predict_charindex == gold_charindex:
                predict_charindex += len(predict_word[predict_index])
                gold_charindex += len(gold_word[gold_index])
                predict_index += 1
                gold_index += 1
            elif predict_charindex<gold_charindex:
                predict_charindex += len(predict_word[predict_index])
                predict_index+=1
            elif predict_charindex>gold_charindex:
                gold_charindex+=len(gold_word[gold_index])
                gold_index+=1

    return seg_correct_num,joint_correct_num,predict_all_num,gold_all_num

class Action(Enum):
    APPEND = 0
    SEPARATE = 1
    No = 2


class State(object):

    def __init__(self, word=None, tag=None, isGold=False,whole_s=None):
        if whole_s is None:
            print('whole_s should be not None,exit!')
            exit()
        self.whole_s = whole_s
        if word is None:
            word = []
        if tag is None:
            tag = []
        self.tag = copy.deepcopy(tag)
        self.word = copy.deepcopy(word)
        self.score = 0
        assert len(tag) == len(word)
        self.charLen = sum(map(lambda x: len(x), word))
        self.word.insert(0, ' ')
        self.tag.insert(0, 'PAD')
        self.word.insert(1, ' ')
        self.tag.insert(1, 'PAD')
        if isGold:
            self.word.append(' ')
            self.tag.append('PAD')
            self.charLen+=1

    def __eq__(self, other):
        for i in range(len(self.word)):
            if self.word[i] != other.word[i] or self.tag[i] != other.tag[i]:
                return False
            # try:
            #     if self.word[i] != other.word[i] or self.tag[i] != other.tag[i]:
            #         return False
            # except IndexError as e:
            #     print(self.word)
            #     print(self.tag)
            #     print(other.word)
            #     print(other.tag)
        return True

    def follow(self, gold):
        l = self.nowLen()
        wl = self.nowWordLen()
        # print('l:', l)
        # if l == 0:
        #     self.word.append(novel.word[0][0])
        #     self.tag.append(novel.tag[0])
        #     self.charLen += 1
        #     return
        if len(gold.word[l - 1]) > wl:
            self.word[-1] = gold.word[l - 1][0:wl + 1]
            self.charLen += 1
            return Action.APPEND
        elif len(gold.word[l - 1]) == wl:
            self.word.append(gold.word[l][0])
            self.tag.append(gold.tag[l])
            self.charLen += 1
            return Action.SEPARATE

    def nowLen(self):  # num of words
        return len(self.word)

    def nowWordLen(self):  # len of now word
        return len(self.word[-1])

    def append(self, c):
        new_state = copy.deepcopy(self)
        new_state.charLen += 1
        new_state.word[-1] += c
        return new_state

    def separate(self, c, t):
        new_state = copy.deepcopy(self)
        new_state.word.append(c)
        new_state.tag.append(t)
        new_state.charLen += 1
        return new_state


class AgendaBeam(object):
    def __init__(self, n=16):
        self.old = []
        self.new = []

    def update(self):
        self.old = self.new
        self.new.clear()

    def squeeze(self, n):
        self.old.sort(key=lambda x: x.score, reverse=True)
        if n > len(self.old):
            n = len(self.old)
        signatures = {}
        self.old = self.old[0:n]
        for old_state in self.old:
            if signatures.get(old_state.word[-2] + '_' + old_state.tag[-2] + '_' + old_state.tag[-1]) is None:
                signatures[old_state.word[-2] + '_' + old_state.tag[-2] + '_' + old_state.tag[-1]] = old_state
            else:
                if signatures[
                    old_state.word[-2] + '_' + old_state.tag[-2] + '_' + old_state.tag[-1]].score < old_state.score:
                    signatures[old_state.word[-2] + '_' + old_state.tag[-2] + '_' + old_state.tag[-1]] = old_state.score
        self.old = list(signatures.values())
        self.old.sort(key=lambda x: x.score, reverse=True)
        # print('len after squeeze: ',len(self.old))

    def clear(self):
        self.old = []
        self.new = []

class Weight(object):
    def __init__(self,i_round,value=0):
        self.now = value
        self.accumulated = 0
        self.used = 0
        self.last_update = i_round
    def accumulate(self,i_round):
        self.accumulated+=(i_round-self.last_update)*self.now
        self.last_update = i_round

    def update(self,amount,i_round):
        self.accumulate(i_round)
        self.now+=amount

    def useAverage(self,n_iterations):
        self.used = self.accumulated/n_iterations

    def useRaw(self):
        self.used = self.now

class FeatureWeight(object):
    # [newest_weight,weight_sum,weight_tmp_avg]
    def __init__(self):
        self.weightDict = {}

    def getFeatureScore(self, feature, isTrain):
        r = self.weightDict.get(feature)
        if r is None:
            return 0
        if isTrain:
            # print('train')
            return r.now
        else:
            # print('not train')
            return r.used

    def updateFeatureScore(self, feature, amount, i_round):
        weight = self.weightDict.get(feature)
        if weight:
            # if feature == 't1:（' or feature == 't1:）':
            #     print('update',feature,amount)
            weight.update(amount,i_round)
        else:
            # print('set new', feature, amount)
            self.weightDict[feature] = Weight(i_round,amount)
            # if feature == 't1:（' or feature == 't1:）':
            #     print('create',feature,amount)

    def accumulateAll(self,i_round):
        for w in self.weightDict.values():
            w.accumulate(i_round)

    def useRaw(self):
        for w in self.weightDict.values():
            w.useRaw()

    def useAverage(self,n_iterations):
        for w in self.weightDict.values():
            w.useAverage(n_iterations)


    # def accumulateWeight(self):
    #     for value in self.weightDict.values():
    #         value[1]+=value[0]


    # def getAverageWeight(self,iterations):
    #     for value in self.weightDict.values():
    #         value[2] = value[1]/iterations



class EnhancedTagger(object):
    def __init__(self,args=None):
        self.args = args
        self.agenda = AgendaBeam(16)
        self.weight = FeatureWeight()
        self.word2tags = {}
        self.char2tags = {}
        tmp = set()
        tmp.add('PAD')
        self.char2tags[' '] = tmp
        self.char2tags_hash = {}
        self.frequent_word = set()
        self.word_frequency = {}
        self.max_frequency = 0
        self.tag2length = {}
        self.tag_set = set()
        # self.tag_set.add('PAD')
        self.train_rules = []
        self.char_can_start = set()
        self.threshold = 100
        self.tripples = set()
        self.pku_dict = set()
        self.pattern2num = dict()
        self.noun2pattern = dict()
        # self.PENN_TAG_CLOSED = {'P','DEC','DEG','CC','LC','PN','DT','VC','AS','VE','ETC','MSP','CS','BA','DEV','SB','SP','LB','DER','PU'}
        # self.PEOPLE_TAG_CLOSED = {'p',}
        # self.PENN_TAG_CLOSED = set()

        self.train_w = set()
        self.train_p = set()
        self.test_w = set()
        self.test_p = set()
        if args:
            if args.dataset_train == 'pku':
                if self.args.use_closed_set=='1':
                    # self.PENN_TAG_CLOSED = {'p', 'c', 'f', 'r', 'uv', 'y', 'w'}
                    self.PENN_TAG_CLOSED = {'w','u','p','c','f','r','q','y','e','o','l','i','k','h'}
                else:
                    self.PENN_TAG_CLOSED = set()
                self.noun_tag = {'n', 'nz', 'ns', 'nr', 'nt'}
            elif args.dataset_train == 'ctb':
                self.PENN_TAG_CLOSED = {'p','dec','deg','c','lc','pn','dt','vc','as','ve','etc','msp','cs','ba','dev','sb','sp','lb','der','pu'}
                self.noun_tag = {'nr', 'nt', 'nn'}
            else:
                print('non-support dataset!')
                exit(2)
        # due to the loss of knowledge about linguistics, I can't transform the tag closed
        # set into tag set in PeopleDaily, so the prune is'nt implemented here
        self.firstChar2tag = {}
    def prepareKnowledge(self, training_set):
        f = open(self.args.pku_dict,'r',encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            self.pku_dict.add(line.split('\t')[0])
        print(self.pku_dict)
        print('above is pku_dict')
        states = []
        for i in range(len(training_set[0])):
            states.append(State(training_set[0][i], training_set[1][i], True,whole_s=training_set[2][i]))

        # self.wc = WordClassifier(self.noun_tag)
        # self.wc.prepareKnowledge(states)
        # self.wc.prepareData(states)
        # self.pc = POSClassifier(self.wc.positive_candidate,self.wc.positive2tags,self.noun_tag)
        # self.pc.prepareData(states)
        # self.wc.train()
        # self.pc.train(21)
        # self.pc.weight.accumulateAll(21)
        # self.pc.weight.useAverage(21)

        for i in range(len(states)):
            for j in range(2,len(states[i].word)-1):
                if states[i].tag[j] in self.noun_tag:
                    now_pattern = states[i].word[j-1]+'-'+states[i].word[j+1]
                    self.pattern2num[now_pattern] = \
                        self.pattern2num.setdefault(now_pattern,0)+1

        # print(self.pattern2num.items())
        num10 = 0
        self.P = set()
        self.W = set()
        for item in self.pattern2num.items():
            if item[1]>9:
                self.P.add(item[0])

        for i in range(len(states)):
            for j in range(2,len(states[i].word)-1):
                if states[i].tag[j] in self.noun_tag and \
                    (states[i].word[j-1]+'-'+states[i].word[j+1]) in self.P:
                    pattern_set = self.noun2pattern.get(states[i].word[j])
                    if pattern_set is not None:
                        pattern_set.add(states[i].word[j-1]+'-'+states[i].word[j+1])
                    else:
                        self.noun2pattern[states[i].word[j]] = \
                            {states[i].word[j-1]+'-'+states[i].word[j+1]}

        for w,p_s in self.noun2pattern.items():
            if len(p_s)>1 and w not in self.pku_dict:
                self.W.add(w)

        if ' ' in self.W:
            print('W里有空格，已去除')
            self.W.remove(' ')

        self.train_p = self.P
        self.train_w = self.W

        if args.dataset_test == 'novel':
            w_f = open(args.test_w,'r',encoding='utf-8')
            lines = w_f.readlines()
            for line in lines:
                line = line.strip()
                self.test_w.add(line)

            p_f = open(args.test_p,'r',encoding='utf-8')
            lines = p_f.readlines()
            for line in lines:
                line = line.strip()
                self.test_p.add(line)

        self.test_p = self.test_p.union(self.P)
        self.test_w = self.test_w.union(self.W)

        print(list(self.P)[:40])
        # print(len(self.P))

        print(list(self.W)[:40])
        # print(len(self.W))



        for i in range(len(training_set[0])):
            for j in range(len(training_set[0][i])):
                tmp = self.word2tags.get(training_set[0][i][j])
                if tmp is None:
                    tmp = self.word2tags[training_set[0][i][j]] = set()
                    tmp.add(training_set[1][i][j])
                else:
                    tmp.add(training_set[1][i][j])
                for c in training_set[0][i][j]:
                    tmp = self.char2tags.get(c)
                    if tmp is None:
                        tmp = self.char2tags[c] = set()
                        tmp.add(training_set[1][i][j])
                    else:
                        tmp.add(training_set[1][i][j])
                tmp = self.firstChar2tag.get(training_set[0][i][j][0])
                if tmp:
                    tmp.add(training_set[1][i][j])
                else:
                    self.firstChar2tag[training_set[0][i][j][0]] = {training_set[1][i][j]}

                # self.word2tags[training_set[0][i][j]] = training_set[1][i][j]
                self.char_can_start.add(training_set[0][i][j][0])
                if self.word_frequency.get(training_set[0][i][j]) is None:
                    self.word_frequency[training_set[0][i][j]] = 1
                else:
                    self.word_frequency[training_set[0][i][j]] += 1

                if self.tag2length.get(training_set[1][i][j]) is None:
                    self.tag_set.add(training_set[1][i][j])
                    self.tag2length[training_set[1][i][j]] = len(training_set[0][i][j])
                elif self.tag2length[training_set[1][i][j]] < len(training_set[0][i][j]):
                    self.tag2length[training_set[1][i][j]] = len(training_set[0][i][j])

            self.train_rules.append(self.judge_by_rule(training_set[2][i]))

        threshold = int(max(self.word_frequency.values()) / 5000) + 5
        self.threshold = threshold
        for pair in self.word_frequency.items():
            if pair[1] > threshold:
                self.frequent_word.add(pair[0])
        self.tag_list = list(self.tag_set)
        self.tag_list.sort()
        self.hash_char2tags()
        print('prepare knowledge successfully')

    def canStart(self,sentence,index,tag):
        if self.args.dataset_train =='pku':
            CD = 'm'
        elif self.args.dataset_train == 'ctb':
            CD = 'cd'
        if tag not in self.PENN_TAG_CLOSED and tag!=CD :
            return True
        else:
            # print(tag)
            # print(index)
            # print(sentence[index])
            if tag in self.firstChar2tag.setdefault(sentence[index],set()):
                if tag==CD:
                    return True
                for j in range(1, self.tag2length[tag] + 2):
                    if self.word2tags.get(sentence[index:index + j]) and tag in self.word2tags[sentence[index:index + j]]:
                        return True
                return False
            else:
                return False

    def hash_tags(self,tags):
        result = ['0']*(len(self.tag_list)+1)# PAD not in tag_set
        for i in range(len(self.tag_list)):
            if self.tag_list[i] in tags:
                result[i] = '1'
        if 'PAD' in tags:
            result[-1] = '1'
        return ''.join(result)

    def hash_char2tags(self):
        for pair in self.char2tags.items():
            self.char2tags_hash[pair[0]] = self.hash_tags(pair[1])


    def judge_by_rule(self, sentence):

        return [Action.SEPARATE] + [Action.No] * (len(sentence) - 1)


        actions = []
        actions.append(Action.SEPARATE)
        lastChar = sentence[0]

        def notChinese(c):
            if c.encode('utf-8').isalnum():
                return True
            if c in {'&','-','\''}:
                return True
            return False
        for i in range(1, len(sentence)):
            lastChar = sentence[i-1]
            newChar = sentence[i]
            n1 = notChinese(lastChar)
            n2 = notChinese(newChar)
            if n1 and n2:
                actions.append(Action.APPEND)
            elif n1 == n2:
                actions.append(Action.No)
            else:
                actions.append(Action.SEPARATE)

        return actions

    def tag(self, sentence, isTrain, rule, i_round=None,gold=None,sentence_index = 0,isDebug=False,f=None,showPredictProcess=False):
        s = sentence
        self.agenda.old = [State(whole_s=sentence)]
        goldFollow = State(whole_s=sentence)
        for i in range(len(sentence)):
            if len(self.agenda.old) == 0:
                print('hasnot state return 1 at index',i)
                print()
                return 1
            anyCorrect = False
            # try:
            #     assert len(self.agenda.new) == 0
            # except AssertionError
            if isTrain:
                for state in self.agenda.old:
                    if state == goldFollow:
                        anyCorrect = True
                        break

                if not anyCorrect:
                    # if '（' in goldFollow.word or ')' in goldFollow.word:
                    #     print('early stop')
                    self.updateScoreForState(self.agenda.old[0], -1,i_round)
                    self.updateScoreForState(goldFollow, 1,i_round)
                    if isDebug:
                        f.write(str(sentence_index)+'th s,early update return 1\n')
                        print(sentence_index,'th s,early update return 1')
                        f.write('predict:\n')
                        print('predict:\n',end='')
                        # try:
                        for w in self.agenda.old[0].word:
                            f.write(w+'-')
                            print(w,end='-')
                        print('')
                        f.write('\n')
                        for t in self.agenda.old[0].tag:
                            f.write(t+'-')
                            print(t,end='-')
                        print('')
                        f.write('\n')
                        f.write('goldFollow:\n')
                        print('goldFollow:\n',end='')
                        for w in goldFollow.word:
                            f.write(w+'-')
                            print(w,end='-')
                        print('')
                        f.write('\n')
                        for t in goldFollow.tag:
                            f.write(t+'-')
                            print(t,end='-')
                        print('')
                        f.write('\n')
                        f.write('novel:\n')
                        print('novel:\n',end='')
                        for w in gold.word:
                            f.write(w+'-')
                            print(w,end='-')
                        print('')
                        f.write('\n')
                        for t in gold.tag:
                            f.write(t+'-')
                            print(t,end='-')
                        f.write('\n')
                        print('')
                    self.agenda.clear()
                    return 1

            if rule[i] == Action.No:
                for old_state in self.agenda.old:
                    # if i==len(sentence)-1:
                    #     print(old_state.word)
                    #     print(sentence[i] in self.char_can_start)
                    #     print(self.canAssignTag(old_state.word[-1], old_state.tag[-1]))

                    if  self.tag2length[old_state.tag[-1]] > len(old_state.word[-1]):
                        appended = old_state.append(sentence[i])
                        appended.score += self.getOrUpdateAppendScore(appended,isTrain)
                        self.agenda.new.append(appended)
                    if sentence[i] in self.char_can_start and (i == 0 or self.canAssignTag(old_state.word[-1], old_state.tag[-1])):
                        for tag in self.tag_set:
                            canStart = self.canStart(sentence, i, tag)
                            if not canStart:
                                continue
                            separated = old_state.separate(sentence[i], tag)
                            separated.score += self.getOrUpdateSeparateScore(separated,isTrain)
                            self.agenda.new.append(separated)

            elif rule[i] == Action.SEPARATE:
                for old_state in self.agenda.old:
                    for tag in self.tag_set:
                        canStart = self.canStart(sentence,i,tag)
                        if not canStart:
                            continue
                        separated = old_state.separate(sentence[i], tag)
                        separated.score += self.getOrUpdateSeparateScore(separated,isTrain)
                        self.agenda.new.append(separated)
            elif rule[i] == Action.APPEND:
                for old_state in self.agenda.old:
                    appended = old_state.append(sentence[i])
                    appended.score += self.getOrUpdateAppendScore(appended,isTrain)
                    self.agenda.new.append(appended)
            if len(self.agenda.new) == 0:
                print(rule[i])
                print(self.agenda.old[0].word)
                print(self.agenda.old[0].tag)
                print('\n',i)

                if self.args.record:
                    f_error = open(args.record + '/error_record.txt', 'a', encoding='utf-8')
                    print(rule[i],file=f_error)
                    print(self.agenda.old[0].word,file=f_error)
                    print(self.agenda.old[0].tag,file=f_error)
                    print('\n', i,file=f_error)

                if isTrain:
                    for old_state in self.agenda.old:
                        # if i==len(sentence)-1:
                        #     print(old_state.word)
                        #     print(sentence[i] in self.char_can_start)
                        #     print(self.canAssignTag(old_state.word[-1], old_state.tag[-1]))

                        if True:
                            appended = old_state.append(sentence[i])
                            appended.score += self.getOrUpdateAppendScore(appended, isTrain)
                            self.agenda.new.append(appended)
                        if True:
                            for tag in self.tag_set:
                                separated = old_state.separate(sentence[i], tag)
                                separated.score += self.getOrUpdateSeparateScore(separated, isTrain)
                                self.agenda.new.append(separated)
                    # max_index = 0
                    # for i in range(len(self.agenda.old)):
                    #     self.agenda.old[i] = self.agenda.old[i].separate(' ', 'PAD')
                    #     self.agenda.old[i].score += self.getOrUpdateSeparateScore(self.agenda.old[i], isTrain)
                    #     if self.agenda.old[i].score > self.agenda.old[max_index].score:
                    #         max_index = i
                    #
                    # self.updateScoreForState(self.agenda.old[max_index], -1, i_round)
                    # self.updateScoreForState(novel, 1, i_round)

                else: #if in test and all state cut by pruning, just don't use prune temporarily
                    for old_state in self.agenda.old:
                        # if i==len(sentence)-1:
                        #     print(old_state.word)
                        #     print(sentence[i] in self.char_can_start)
                        #     print(self.canAssignTag(old_state.word[-1], old_state.tag[-1]))

                        if True:
                            appended = old_state.append(sentence[i])
                            appended.score += self.getOrUpdateAppendScore(appended, isTrain)
                            self.agenda.new.append(appended)
                        if True:
                            for tag in self.tag_set:
                                separated = old_state.separate(sentence[i], tag)
                                separated.score += self.getOrUpdateSeparateScore(separated, isTrain)
                                self.agenda.new.append(separated)


            self.agenda.old = self.agenda.new
            self.agenda.new = []
            # print('old:')
            # for w in self.agenda.old:
            #     print(w.word)
            self.agenda.squeeze(16)
            if isTrain:
                goldFollow.follow(gold)

        max_index = 0
        for i in range(len(self.agenda.old)):
            self.agenda.old[i] = self.agenda.old[i].separate(' ', 'PAD')
            self.agenda.old[i].score += self.getOrUpdateSeparateScore(self.agenda.old[i],isTrain)
            if self.agenda.old[i].score > self.agenda.old[max_index].score:
                max_index = i

        if isTrain:
            if self.agenda.old[max_index] == gold:
                if isDebug:
                    f.write('predict right return 0\n')
                    print('predict right return 0',sentence)
                    f.write('predict:\n')
                    print('predict:\n', end='')
                    # try:
                    for w in self.agenda.old[max_index].word:
                        f.write(w + '-')
                        print(w, end='-')
                    f.write('\n')
                    print('')
                    for t in self.agenda.old[max_index].tag:
                        f.write(t + '-')
                        print(t, end='-')
                    print('')
                    f.write('\n')
                    f.write('novel:\n')
                    print('novel:\n', end='')
                    for w in gold.word:
                        f.write(w + '-')
                        print(w, end='-')
                    print('')
                    f.write('\n')
                    for t in gold.tag:
                        f.write(t + '-')
                        print(t, end='-')
                    f.write('\n')
                    print('')
                return 0
            else:
                # if '（' in novel.word or ')' in novel.word:
                #     print('predict wrong')
                self.updateScoreForState(self.agenda.old[max_index], -1, i_round)
                self.updateScoreForState(gold, 1,i_round)
                if isDebug:
                    f.write('predict wrong return 1\n')
                    print('predict wrong return 1')
                    f.write('predict:\n')
                    print('predict:\n', end='')
                    # try:
                    for w in self.agenda.old[max_index].word:
                        f.write(w + '-')
                        print(w, end='-')
                    f.write('\n')
                    print('')
                    for t in self.agenda.old[max_index].tag:
                        f.write(t + '-')
                        print(t, end='-')
                    print('')
                    f.write('\n')
                    f.write('novel:\n')
                    print('novel:\n', end='')
                    for w in gold.word:
                        f.write(w + '-')
                        print(w, end='-')
                    f.write('\n')
                    print('')
                    for t in gold.tag:
                        f.write(t + '-')
                        print(t, end='-')
                    f.write('\n')
                    print('')
                self.agenda.clear()
                return 1
        else:
            return self.agenda.old[max_index]

    def updateScoreForState(self, state, amount,i_round):
        tmp_state = State(whole_s=state.whole_s)
        # if '（' in state.word or '）' in state.word:
        #     print(state.word)
        #     print(state.tag)
        for i in range(state.charLen):
            action = tmp_state.follow(state)
            if action == Action.APPEND:
                self.getOrUpdateAppendScore(tmp_state,True, amount,i_round)
            elif action == Action.SEPARATE:
                self.getOrUpdateSeparateScore(tmp_state,True, amount,i_round)

        if tmp_state != state:
            raise ('follow not finished!')

    def getOrUpdateSeparateScore(self, state,isTrain,amount=0,i_round=0):
        # print(state.word[-1])
        if isTrain:
            self.W = self.train_w
            self.P = self.train_p
        else:
            self.W = self.test_w
            self.P = self.test_p
        assert len(state.word[-1]) == 1
        features = []
        features.append(seenWord(state))
        features.append(lastWordByWord(state))
        if len(state.word[-2]) == 1:
            features.append(oneCharWord(state))
        features.append(lengthByFirstChar(state))
        features.append(lengthByLastChar(state))
        features.append(separateChars(state))
        features.append(firstAndLastChars(state))
        features.append(lastWordFirstChar(state))
        features.append(currentWordLastChar(state))
        features.append(firstCharLastWordByWord(state))
        features.append(lastWordByLastChar(state))
        features.append(lengthByLastWord(state))
        features.append(lastLengthByWord(state))
        features.append(currentTag(state))
        features.append(lastTagByTag(state))
        features.append(lastTwoTagsByTag(state))
        features.append(tagByLastWord(state))
        features.append(lastTagByWord(state))
        features.append(tagByWordAndPrevChar(state))
        features.append(tagByWordAndNextChar(state))
        features.append(tagByFirstChar(state))
        features.append(tagByLastChar(state))
        features.append(tagByChar(state))
        features.extend(taggedCharByLastChar(state))  # return a list
        features.append(taggedSeparateChars(state))
        features.append(tagByFirstCharCat(state,self.char2tags_hash.setdefault(state.word[-1],'none')))
        features.append(tagByLastCharCat(state,self.char2tags_hash.setdefault(state.word[-2][-1],'none')))
        #以下为新加的特征
        if self.args.new_feature:
            features.append(wordTagTag(state))
            features.append(tagWordTag(state))
            features.append(firstCharBy2Tags(state))
            features.append(firstCharBy3Tags(state))
            features.append(tag0Tag1Size1(state))
            features.append(tag1Tag2Size1(state))
            features.append(tag0Tag1Tag2Size1(state))

        #以下为enhanced tagger的特征
        if state.word[-3] in self.W:
            features.append(isNoun2Tag2Len2(state))

        if state.word[-2] in self.W:
            features.append(isNoun1Tag1Len1(state))



        is_noun_candidate = False
        # tmp_s = state.word[-1]
        now_w0_len = 0
        # print('word')
        # print(state.whole_s)
        for i in range(1,5):
            # print(state.word[-1])
            # print(state.whole_s[state.charLen-1:state.charLen-1+i])
            if state.whole_s[state.charLen-1:state.charLen-1+i] in self.W:
                # print('被选中为noun')
                now_w0_len = i
                is_noun_candidate = True
                break
        if is_noun_candidate:
            features.append(isNoun0Tag0Len0(state,now_w0_len))


        # tmp_is_tripple = False
        tmp_is_pattern = False
        # print('pattern')
        # print(state.whole_s)

        if self.args.use_pattern_feature and len(state.word)>1:
            for i in range(1,5):
                # print(state.word[-1])
                # print(state.word[-3]+'-'+state.whole_s[state.charLen-1:state.charLen-1+i])
                if state.word[-3]+'-'+state.whole_s[state.charLen-1:state.charLen-1+i] in self.P:
                    tmp_is_pattern = True
                    # print('被选中为pattern')
                    break

        if tmp_is_pattern:
            features.append(ispatternTag1Len1(state))
            if state.word[-2] in self.W:
                features.append(istrippleTag1Len1(state))
        # features.append(ispatternTag1Len1(state,self.ispatternSet))
        # features.append(istrippleTag1Len1_1(state,self.istrippleSet,self.W))
        if amount == 0:
            score = 0
            for feature in features:
                score += self.weight.getFeatureScore(feature,isTrain)
            return score
        else:
            for feature in features:
                self.weight.updateFeatureScore(feature, amount,i_round)

    def getOrUpdateAppendScore(self, state,isTrain, amount=0,i_round=0):
        features = []
        features.append(consecutiveChars(state))
        features.append(tagByChar(state))
        features.append(taggedCharByFirstChar(state))
        features.append(taggedConsecutiveChars(state))
        if amount == 0:
            score = 0
            for feature in features:
                score += self.weight.getFeatureScore(feature,isTrain)
            return score
        else:
            for feature in features:
                self.weight.updateFeatureScore(feature, amount,i_round)

    def canAssignTag(self, word, tag):
        if word not in self.frequent_word and tag not in self.PENN_TAG_CLOSED:
            return True
        else:
            if tag in self.word2tags.setdefault(word,set()):
                return True
            else:
                return False

    def train(self, training_set,start_epoch, iterations,isDebug=False,test_set=None,dev_set=None,args=None):
        test_gold_state = []
        if test_set:
            for i in range(len(test_set[0])):
                test_gold_state.append(State(test_set[0][i],test_set[1][i],isGold=True,whole_s=test_set[2][i]))
        # dev_gold_state = []
        # if dev_set:
        #     for i in range(len(dev_set[0])):
        #         dev_gold_state.append(State(dev_set[0][i],dev_set[1][i],isGold=True))
        f_training_process = open('training_record.txt','a',encoding='utf-8')

        if args.record:
            f_error = open(args.record + '/error_record.txt', 'a', encoding='utf-8')
            f_error.write('\n'+'*'*20+'\n')
            f_error.close()
        # if start_epoch ==0 and test_set:
        #     result = self.test(test_set)
        #     f_error.write('untrained weight(all zero) result: ' + str(result) + '\n')
        #     self.weight.accumulateAll(0)
        #     self.weight.useRaw()
        #     result = self.test(test_set)
        #     # f_error.write('unaveraged weight result: ' + str(result) + '\n')
        #     # print('unaveraged weight result: ' + str(result))
        #     self.weight.useAverage(1)
        #     result = self.test(test_set)
        #     print("no_bug!")
        # f_error.write('unaveraged weight result: ' + str(result) + '\n')
        # print('unaveraged weight result: ' + str(result))
        n_error = 0
        golds = []
        for i in range(len(training_set[2])):
            golds.append(State(training_set[0][i], training_set[1][i], True,whole_s=training_set[2][i]))
        old_time = time.time()
        for j in range(start_epoch,start_epoch+iterations):
            old_time = time.time()
            n_error = 0
            if args.weight:
                f_weight = open(args.weight+'/trained_weight_'+str(j)+'.pkl', 'wb')
            f_training_process = open('training_record.txt', 'a',encoding='utf-8')
            for i in range(len(training_set[2])):
                n_error += self.tag(training_set[2][i], True, self.train_rules[i],j,
                                    gold=golds[i],sentence_index=i,isDebug=isDebug,f=f_training_process)
            new_time = time.time()
            if args.record:
                f_error = open(args.record + '/error_record.txt', 'a', encoding='utf-8')
                f_error.write(str(j)+'th error: '+str(n_error)+' / '+str(len(training_set[2]))+'\n')
                f_error.write('training_time_spent:' + str(new_time - old_time) + '\n')
            print(j, 'th error: ', n_error, ' / ', len(training_set[2]))
            print('training time_spent:',new_time-old_time)
            old_time = new_time
            if test_set and j>=int(args.start_test) and j%1==0:
                print("test_set:")
                if args.record:
                    f_error.write('test_set:\n')
                self.weight.useRaw()
                result = self.test(test_set[2],test_gold_state)
                if args.record:
                    f_error.write('unaveraged weight result: '+str(result)+'\n')
                # f_unaveraged = open('unaveraged_weight.pkl','wb')
                # pickle.dump(self.weight.weightDict,f_unaveraged)
                print('unaveraged weight result: '+str(result))
                self.weight.accumulateAll(j + 1)
                self.weight.useAverage(j+1)
                # f_averaged = open('averaged_weight.pkl','wb')
                # pickle.dump(self.weight.weightDict,f_averaged)
                # f_averaged.close()
                # f_unaveraged.close()
                # print('two f for debug save !')
                result = self.test(test_set[2],test_gold_state)
                if self.args.record:
                    f_error.write('averaged weight result: '+str(result)+'\n')
                print('averaged weight result: '+str(result))
                new_time  = time.time()
                print('test time_spent: ',new_time-old_time)

            if args.weight:
                pickle.dump(self.weight.weightDict, f_weight)
            # if dev_set:
            #     self.weight.useRaw()
            #     result = self.test(dev_set[2],dev_gold_state)
            #     f_error.write('unaveraged weight result: '+str(result)+'\n')
            #     # f_unaveraged = open('unaveraged_weight.pkl','wb')
            #     # pickle.dump(self.weight.weightDict,f_unaveraged)
            #     print('unaveraged weight result: '+str(result))
            #     self.weight.accumulateAll(j + 1)
            #     self.weight.useAverage(j+1)
            #     # f_averaged = open('averaged_weight.pkl','wb')
            #     # pickle.dump(self.weight.weightDict,f_averaged)
            #     # f_averaged.close()
            #     # f_unaveraged.close()
            #     # print('two f for debug save !')
            #     result = self.test(dev_set[2],dev_gold_state)
            #     f_error.write('averaged weight result: '+str(result)+'\n')
            #     print('averaged weight result: '+str(result))
            old_time = new_time
            if args.record:
                f_error.close()

    def test(self,test_sentence,gold_state):
        seg_correct_num = 0
        joint_correct_num = 0
        predict_all_num = 0
        gold_all_num = 0
        for i in range(len(test_sentence)):
            predict = self.tag(test_sentence[i],False,self.judge_by_rule(test_sentence[i]))
            # print('predict word:',predict.word)
            # print('predict tag',predict.tag)
            # print('novel word',novel.word)
            # print('novel tag',novel.tag)
            if type(predict) == int:
                print('shit,return int when test')
                continue
            s_c_n,j_c_n,p_a_n,g_a_n = countCorrect(predict,gold_state[i])
            seg_correct_num+=s_c_n
            joint_correct_num+=j_c_n
            predict_all_num+=p_a_n
            gold_all_num+=g_a_n
        # print('all kind num',seg_correct_num,joint_correct_num,predict_all_num,gold_all_num)

        seg_precision = seg_correct_num/predict_all_num
        seg_recall = seg_correct_num/gold_all_num
        seg_f_score = 2/(1/seg_precision+1/seg_recall)
        joint_precision = joint_correct_num/predict_all_num
        joint_recall = joint_correct_num/gold_all_num
        joint_f_score = 2/(1/joint_precision+1/joint_recall)
        result = {}
        result['seg_precision'] = seg_precision
        result['seg_recall'] = seg_recall
        result['seg_f_score'] = seg_f_score
        result['joint_precision'] = joint_precision
        result['joint_recall'] = joint_recall
        result['joint_f_score'] = joint_f_score
        return result

    # def test(self,test_set):
    #     seg_correct_num = 0
    #     joint_correct_num = 0
    #     predict_all_num = 0
    #     gold_all_num = 0
    #     for i in range(len(test_set[2])):
    #         predict = self.tag(test_set[2][i],False,self.judge_by_rule(test_set[2][i]))
    #         # print('predict word:',predict.word)
    #         # print('predict tag',predict.tag)
    #         # print('novel word',novel.word)
    #         # print('novel tag',novel.tag)
    #         s_c_n,j_c_n,p_a_n,g_a_n = countCorrect(predict,State(test_set[0][i],test_set[1][i],isGold=True))
    #         seg_correct_num+=s_c_n
    #         joint_correct_num+=j_c_n
    #         predict_all_num+=p_a_n
    #         gold_all_num+=g_a_n
    #     # print('all kind num',seg_correct_num,joint_correct_num,predict_all_num,gold_all_num)
    #
    #     seg_precision = seg_correct_num/predict_all_num
    #     seg_recall = seg_correct_num/gold_all_num
    #     seg_f_score = 2/(1/seg_precision+1/seg_recall)
    #     joint_precision = joint_correct_num/predict_all_num
    #     joint_recall = joint_correct_num/gold_all_num
    #     joint_f_score = 2/(1/joint_precision+1/joint_recall)
    #     result = {}
    #     result['seg_precision'] = seg_precision
    #     result['seg_recall'] = seg_recall
    #     result['seg_f_score'] = seg_f_score
    #     result['joint_precision'] = joint_precision
    #     result['joint_recall'] = joint_recall
    #     result['joint_f_score'] = joint_f_score
    #     return result





if __name__ == '__main__':
    # t = Tagger()
    # s = '我们打架B&G,P-9'
    # print(s)
    # print(t.judge_by_rule(s))
    # t = Tagger()
    # print(t.judge_by_rule('新华社天津九月一日电（记者刘庆禄、窦合义）'))

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', required=True, help='training set')
    parser.add_argument('--test', default=None, help='test set')
    parser.add_argument('--record', default=None, help='file output to record measure')
    parser.add_argument('--weight', default=None, help='weight output to')
    parser.add_argument('--start', default=0, type=int, help='set the round')
    parser.add_argument('--epoch', default=10, type=int, help='as the name')
    parser.add_argument('--init_weight',required=False,default=None,help='initial feature weight')
    parser.add_argument('--dataset_train',help='ctb or pku')
    parser.add_argument('--dataset_test',help='ctb or pku or novel')
    parser.add_argument('--mode',help='train or test')
    parser.add_argument('--start_test',default=10,help='the epoch after which start test')
    parser.add_argument('--pku_dict',help='the common word in pku dict')
    parser.add_argument('--add',default=None)
    parser.add_argument('--new_feature',default='ri')
    parser.add_argument('--data_seed',default=-1,type=int,help='how to seg data')
    parser.add_argument('--use_pattern_feature',default=False)
    parser.add_argument('--use_closed_set',default='0',help='whether to use penn closed set tag')
    parser.add_argument('--test_w',default=None,help='the novel W file')
    parser.add_argument('--test_p',default=None,help='the novel P file')

    args = parser.parse_args()

    # f_add = open(args.record+'/error_record.txt','a',encoding='utf-8')
    print(args.add)
    # f_add.close()

    t1 = EnhancedTagger(args)
    # train,test = loadNewPeopleDailyData(args.train)
    # t1.prepareKnowledge(train)
    if args.init_weight:
        t1.weight.weightDict = pickle.load(open(args.init_weight,'rb'))

    # t1.train(train,args.start,args.epoch,test_set=test,args=args)

    if args.mode =='train':
        if args.dataset_train == 'pku':
            if args.data_seed != -2:
                train_p,test_p = loadQiuPKU(args.train,args.data_seed)
            else:
                train_p = loadQiuPKU(args.train,args.data_seed)
        elif args.dataset_train == 'ctb':
            train_p,dev_p,test_p = loadCTB3Data(args.train)
        else:
            print('non-support dataset!')
            print(len(args.dataset_train))
            print(repr(args.dataset_train))
            exit(2)

        if args.dataset_test=='novel':
            test_p = loadNovelData(args.test)

        t1.prepareKnowledge(train_p)
        print(len(t1.test_w))
        print(len(t1.test_p))
        print(len(t1.train_w))
        print(len(t1.train_p))
        if args.init_weight:
            t1.weight.weightDict = pickle.load(open(args.init_weight,'rb'))

        print('use_pattern_feature:',args.use_pattern_feature)

        t1.train(train_p,args.start,args.epoch,test_set=test_p,args=args)

    elif args.mode =='test':
        if args.dataset_train == 'pku':
            train,test = loadQiuPKU(args.train,args.data_seed)
        elif args.dataset_train == 'ctb':
            train,dev,test = loadCTB3Data(args.train)

        # if args.dataset_test == 'pku':
        #     test = loadNewPeopleDailyData(args.test)
        if args.dataset_test == 'novel':
            test = loadNovelData(args.test)




        t1.prepareKnowledge(train)

        if not args.init_weight:
            print('no init weight, test no meaning')
        else:
            t1.weight.weightDict = pickle.load(open(args.init_weight,'rb'))

        t1.weight.accumulateAll(int(args.start))
        t1.weight.useAverage(int(args.start))

        # test = [test[0][0:400],test[1][0:400],test[2][0:400]]
        test_gold_state = []
        for i in range(len(test[0])):
            test_gold_state.append(State(test[0][i],test[1][i],isGold=True,whole_s=test[2][i]))

        result = t1.test(test[2],test_gold_state)
        print('average:',result)

        t1.weight.useRaw()
        result = t1.test(test[2],test_gold_state)
        print('unaverage:',result)

    elif args.mode == 'tag':
        if args.dataset_train == 'pku':
            train,test = loadQiuPKU(args.train,args.data_seed)
        elif args.dataset_train == 'ctb':
            train,dev,test = loadCTB3Data(args.train)

        t1.prepareKnowledge(train)

        if args.init_weight:
            t1.weight.weightDict = pickle.load(open(args.init_weight,'rb'))
        else:
            print('no init_weight, tag no meaning!')
            exit(2)

        print('load weight finish!')
        t1.weight.accumulateAll(int(args.start))
        t1.weight.useAverage(int(args.start))

        while True:
            tobe_tagged = input()
        #         old_to = tobe_tagged
        #         tobe_tagged = ''.join(parseTagged(tobe_tagged)[0])
        #         if len(tobe_tagged)==0:
        #             tobe_tagged = old_to
        #         tmp_state = t1.tag(tobe_tagged,False,t1.judge_by_rule(tobe_tagged))
        #         print(len(tmp_state.word))
        #         for i in range(len(tmp_state.word)):
        #             print(tmp_state.word[i]+'_'+tmp_state.tag[i]+' ',end='')