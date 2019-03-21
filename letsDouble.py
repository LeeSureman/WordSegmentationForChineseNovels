from base_tagger import *
from enhanced_tagger_1 import *
import argparse
import copy

def onNewPattern(e_t,states,left,right,w_candidate):
    now_pattern = left+'_'+right
    now_pattern0 = left+'_'+right[0]
    e_t.istrippleSet.add(now_pattern0)
    e_t.ispatternSet.add(now_pattern0)
    e_t.P.add(now_pattern)
    for i in range(len(states)):
        for j in range(2,len(states[i].word)-1):
            if states[i].word[j] not in e_t.W and (states[i].word[j]+'0' in w_candidate or states[i].word[j]+'1' in w_candidate):
                # print(1)
                if states[i].word[j-1] == left and states[i].word[j+1] == right:

                    local_pattern = e_t.wc.noun2pattern.get(states[i].word[j])
                    local_pattern_num = e_t.wc.noun2pattern_num.get(states[i].word[j])
                    if local_pattern:
                        local_pattern.add(now_pattern)
                    else:
                        e_t.wc.noun2pattern[states[i].word[j]] = {now_pattern}

                    if local_pattern_num:
                        e_t.wc.noun2pattern_num[states[i].word[j]] += 1
                    else:
                        e_t.wc.noun2pattern_num[states[i].word[j]] = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--train', required=True, help='training set')
    parser.add_argument('--test', default=None, help='if mode is test,it is annotated,if mode is tag,it is raw')
    # parser.add_argument('--record', default=None, help='file output to record measure')
    # parser.add_argument('--weight', default=None, help='weight output to')
    parser.add_argument('--start', default=0, type=int, help='set the round')
    # parser.add_argument('--epoch', default=10, type=int, help='as the name')
    parser.add_argument('--base_weight',required=True,default=None,help='base weight')
    parser.add_argument('--enhanced_weight',required=True,help='enhanced weight')
    parser.add_argument('--dataset_test',help='ctb or pku or novel')
    # parser.add_argument('--mode',help='train or test')
    parser.add_argument('--pku_dict',help='the common word in pku dict',required=True)
    parser.add_argument('--add',default='')
    parser.add_argument('--dataset_train', help='ctb or pku')
    parser.add_argument('--new_feature',default='1')
    parser.add_argument('--record',default=None)
    parser.add_argument('--data_seed',type=int,default=-1)

    parser.add_argument('--mode',help='tag or test')
    parser.add_argument('--output_tagged',help='the path of the tagged sentenced')
    parser.add_argument('--is_raw',default=True,help='the data format when tagging',type=bool)

    args = parser.parse_args()

    b_t = BaseTagger(args)
    e_t = EnhancedTagger(args)

    if args.mode =='test':

        train,test = loadQiuPKU(args.train,args.data_seed)


        novel_test = loadNovelData(args.test)


        b_t.prepareKnowledge(train)
        b_t.weight.weightDict = pickle.load(open(args.base_weight,'rb'))



        # n_count = 0
        genCorpus_word = set()
        for i in range(len(train[0])):
            for j in range(len(train[0][i])):
                genCorpus_word.add(train[0][i][j])

        auto_split = []
        W_candidate = set()
        for i in range(len(novel_test[0])):
            now_state = b_t.tag(novel_test[2][i],False,b_t.judge_by_rule(novel_test[2][i]))
            auto_split.append(now_state)
            for i in range(2,len(now_state.word)-1):
                if now_state.word[i] not in genCorpus_word:
                    W_candidate.add(now_state.word[i]+str(int(now_state.tag[i-1]=='w'and now_state.tag[i+1] =='w')))

        # print(W_candidate)


        e_t.prepareKnowledge(train)
        e_t.weight.weightDict = pickle.load(open(args.enhanced_weight,'rb'))


        e_t.W = set()

        for p in e_t.P:
            a,b = p.split('_')
            onNewPattern(e_t,auto_split,a,b,W_candidate)


        novel_gold_state = []
        for i in range(len(novel_test[0])):
            novel_gold_state.append(State(novel_test[0][i], novel_test[1][i], isGold=True))

        # old_result = e_t.test(novel_test[2],novel_gold_state)

        # print('general know:',old_result)
        W_candidate_copy = copy.deepcopy(W_candidate)
        noun_tag_set = {'ns','nz','n','nr','nt'}
        novel_noun_set = set()
        for i,s in enumerate(test[0]):
            for j,w in enumerate(test[0][i]):
                if test[1][i][j] in noun_tag_set:
                    novel_noun_set.add(test[0][i][j])

        while True:
            n_count = 0
            for w in W_candidate:
                if w[-1] == '1':
                    t = ['w','w','w']
                    s = State(['',w[:-1],''],t,True)
                else:
                    t = ['','','']
                    s = State(['',w[:-1],''],t,True)

                isNewNoun = e_t.wc.tag(s,3)
                # print(w,isNewNoun)

                if isNewNoun>0.5:
                    print(w, isNewNoun)
                    W_candidate_copy.remove(w)
                    e_t.W.add(w[:-1])
                    n_count+=1
                    for i in range(len(auto_split)):
                        for j in range(2,len(auto_split[i].word)-1):
                            if auto_split[i].word[j] == w[:-1]:
                                if auto_split[i].word[j-1]+'_'+auto_split[i].word[j+1] not in e_t.P:
                                    onNewPattern(e_t,auto_split,auto_split[i].word[j-1],
                                                 auto_split[i].word[j+1],W_candidate)
            print('\n************\n')
            if n_count == 0:
                break
            n_count = 0
            W_candidate = copy.deepcopy(W_candidate_copy)

        noun_acc = len(novel_noun_set & e_t.W) / len(e_t.W)

        b_result = b_t.test(novel_test[2],novel_gold_state)

        e_result = e_t.test(novel_test[2],novel_gold_state)

        print('noun acc:',noun_acc)

        print('base tagger:',b_result)
        print('\nenhanced tagger:',e_result)

        # print('w:',e_t.wc.lr.w)
        # print('b:',e_t.wc.lr.b)

    else if args.mode =='tag':
        train,test = loadQiuPKU(args.train,args.data_seed)

        if args.is_raw:
            novel_fp = open(args.test, 'r', encoding='utf-8')
            novel_raw = novel_fp.readlines()

        else:
            novel_test = loadNovelData(args.test)
        # novel_fp = open(args.test,'r',encoding='utf-8')
            novel_raw = novel_test[2]

        for i,line in enumerate(novel_raw):
            novel_raw[i] = line.strip()


        b_t.prepareKnowledge(train)
        b_t.weight.weightDict = pickle.load(open(args.base_weight,'rb'))



        # n_count = 0
        genCorpus_word = set()
        for i in range(len(train[0])):
            for j in range(len(train[0][i])):
                genCorpus_word.add(train[0][i][j])

        auto_split = []
        W_candidate = set()
        for i in range(len(novel_raw)):
            now_state = b_t.tag(novel_raw[i],False,b_t.judge_by_rule(novel_raw[i]))
            auto_split.append(now_state)
            for i in range(2,len(now_state.word)-1):
                if now_state.word[i] not in genCorpus_word:
                    W_candidate.add(now_state.word[i]+str(int(now_state.tag[i-1]=='w'and now_state.tag[i+1] =='w')))

        # print(W_candidate)


        e_t.prepareKnowledge(train)
        e_t.weight.weightDict = pickle.load(open(args.enhanced_weight,'rb'))


        e_t.W = set()

        for p in e_t.P:
            a,b = p.split('_')
            onNewPattern(e_t,auto_split,a,b,W_candidate)


        # novel_gold_state = []
        # for i in range(len(novel_test[0])):
        #     novel_gold_state.append(State(novel_test[0][i], novel_test[1][i], isGold=True))

        # old_result = e_t.test(novel_test[2],novel_gold_state)

        # print('general know:',old_result)
        W_candidate_copy = copy.deepcopy(W_candidate)
        noun_tag_set = {'ns','nz','n','nr','nt'}
        novel_noun_set = set()
        for i,s in enumerate(test[0]):
            for j,w in enumerate(test[0][i]):
                if test[1][i][j] in noun_tag_set:
                    novel_noun_set.add(test[0][i][j])

        while True:
            n_count = 0
            for w in W_candidate:
                if w[-1] == '1':
                    t = ['w','w','w']
                    s = State(['',w[:-1],''],t,True)
                else:
                    t = ['','','']
                    s = State(['',w[:-1],''],t,True)

                isNewNoun = e_t.wc.tag(s,3)
                # print(w,isNewNoun)

                if isNewNoun>0.5:
                    print(w, isNewNoun)
                    W_candidate_copy.remove(w)
                    e_t.W.add(w[:-1])
                    n_count+=1
                    for i in range(len(auto_split)):
                        for j in range(2,len(auto_split[i].word)-1):
                            if auto_split[i].word[j] == w[:-1]:
                                if auto_split[i].word[j-1]+'_'+auto_split[i].word[j+1] not in e_t.P:
                                    onNewPattern(e_t,auto_split,auto_split[i].word[j-1],
                                                 auto_split[i].word[j+1],W_candidate)
            print('\n************\n')
            if n_count == 0:
                break
            n_count = 0
            W_candidate = copy.deepcopy(W_candidate_copy)



        for line in novel_raw:
            tmp_r = e_t.tag(line,False,e_t.judge_by_rule(line))
            print(tmp_r)
        # noun_acc = len(novel_noun_set & e_t.W) / len(e_t.W)

        # b_result = b_t.test(novel_test[2],novel_gold_state)

        # e_result = e_t.test(novel_test[2],novel_gold_state)

        # print('noun acc:',noun_acc)


        # print('base tagger:',b_result)
        # print('\nenhanced tagger:',e_result)


