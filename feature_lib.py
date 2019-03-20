def seenWord(state):
    assert len(state.word[-1]) == 1
    return 't1:' + state.word[-2]


def lastWordByWord(state):
    assert len(state.word[-1]) == 1
    return 't2:' + state.word[-3] + '_' + state.word[-2]


def oneCharWord(state):
    assert len(state.word[-1]) == 1
    assert len(state.word[-2]) == 1
    return 't3:' + state.word[-2]


def lengthByFirstChar(state):
    assert len(state.word[-1]) == 1
    return 't4:' + state.word[-2][0] + '_' + str(len(state.word[-2]))


def lengthByLastChar(state):
    assert len(state.word[-1]) == 1
    return 't5:' + state.word[-2][-1] + '_' + str(len(state.word[-2]))


def separateChars(state):
    assert len(state.word[-1]) == 1
    return 't6:' + state.word[-2][-1] + '_' + state.word[-1][0]


def consecutiveChars(state):
    assert len(state.word[-1]) > 1
    return 't7:' + state.word[-1][-2:]


def firstAndLastChars(state):
    assert len(state.word[-1]) == 1
    return 't8:' + state.word[-2][0] + '_' + state.word[-2][-1]


def lastWordFirstChar(state):
    assert len(state.word[-1]) == 1
    return 't9:' + state.word[-2] + '_' + state.word[-1][0]


def currentWordLastChar(state):
    assert len(state.word[-1]) == 1
    return 't10:' + state.word[-3][-1] + '_' + state.word[-2]


def firstCharLastWordByWord(state):
    assert len(state.word[-1]) == 1
    return 't11:' + state.word[-2][0] + '_' + state.word[-1][0]


def lastWordByLastChar(state):
    assert len(state.word[-1]) == 1
    return 't12:' + state.word[-3][-1] + '_' + state.word[-2][-1]


def lengthByLastWord(state):
    assert len(state.word[-1]) == 1
    return 't13:' + state.word[-3] + '_' + str(len(state.word[-2]))


def lastLengthByWord(state):
    assert len(state.word[-1]) == 1
    return 't14:' + str(len(state.word[-3])) + '_' + state.word[-2]


def currentTag(state):
    assert len(state.word[-1]) == 1
    return 't15:' + state.word[-2] + '_' + state.tag[-2]


def lastTagByTag(state):
    assert len(state.word[-1]) == 1
    try:
        r = 't16:' + state.tag[-2] + '_' + state.tag[-1]
    except TypeError as e:
        print(state.word[-2])
        print(state.word[-1])
        print(state.tag[-2])
        print(state.tag[-1])
    # return 't16:' + state.tag[-2] + '_' + state.tag[-1]
    return r

def lastTwoTagsByTag(state):
    assert len(state.word[-1]) == 1
    return 't17:' + state.tag[-3] + '_' + state.tag[-2] + '_' + state.tag[-1]


def tagByLastWord(state):
    assert len(state.word[-1]) == 1
    return 't18:' + state.word[-2] + '_' + state.tag[-1]


def lastTagByWord(state):
    assert len(state.word[-1]) == 1
    return 't19:' + state.tag[-3] + '_' + state.word[-2]


def tagByWordAndPrevChar(state):
    assert len(state.word[-1]) == 1
    return 't20:' + state.word[-3][-1] + '_' + state.word[-2] + '_' + state.tag[-2]


def tagByWordAndNextChar(state):
    assert len(state.word[-1]) == 1
    return 't21:' + state.word[-2] + '_' + state.tag[-2] + '_' + state.word[-1][0]


def tagOfOneCharWord(state):
    assert len(state.word[-1]) == 1
    assert len(state.word[-2]) == 1
    return 't22:' + state.word[-3][-1] + '_' + state.word[-2] + '_' + state.tag[-2] + '_' + state.word[-1][0]


def tagByFirstChar(state):
    assert len(state.word[-1]) == 1
    return 't23:' + state.word[-1][0] + '_' + state.tag[-1]


def tagByLastChar(state):
    assert len(state.word[-1]) == 1
    return 't24:' + state.word[-2][-1] + '_' + state.tag[-2]


def tagByChar(state):
    #be called either append or separate
    # seems like tagByFirstChar, but when they are called are diffenrent
    return 't25:' + state.word[-1][-1] + '_' + state.tag[-1]

def taggedCharByFirstChar(state):
    assert len(state.word[-1]) > 1
    return 't26:'+state.word[-1][0]+'_'+state.tag[-1]+'_'+state.word[-1][-1]


def taggedCharByLastChar(state):
    result = []
    for i in range(len(state.word[-2])-1):
        result.append('t27:'+state.tag[-2]+'_'+state.word[-2][i]+'_'+state.word[-2][-1])

    return result


def tagByFirstCharCat(state,tagset_hash):
    assert len(state.word[-1]) == 1
    # index = tag2index[state.tag[-1]]
    # tagset_hash = tagset_hash[:index]+'1'+tagset_hash[index+1:]
    return 't28:'+state.tag[-1]+'_'+tagset_hash


def tagByLastCharCat(state,tagset_hash):
    assert len(state.word[-1]) == 1
    # print(tag2index)
    # index = tag2index[state.tag[-2]]
    # tagset_hash = tagset_hash[:index]+'1'+tagset_hash[index+1:]
    return 't29:'+state.tag[-2]+'_'+tagset_hash

def taggedSeparateChars(state):
    assert len(state.word[-1]) == 1
    return 't30:'+state.word[-2][0]+'_'+state.tag[-2]+'_'+state.word[-1][0]+'_'+state.tag[-1]

def taggedConsecutiveChars(state):
    return 't31:'+state.word[-1][-2]+'_'+state.tag[-1]+'_'+state.word[-1][-1]

#以下为新加的，zpar-0.6的agenda的新特征
def wordTagTag(state):
    return 't32:'+state.word[-3]+'_'+state.tag[-2]+'_'+state.tag[-1]

def tagWordTag(state):
    return 't33:'+state.tag[-3]+'_'+state.word[-2]+'_'+state.tag[-1]

def firstCharBy2Tags(state):
    return 't34:'+state.tag[-2]+'_'+state.word[-1][0]+'_'+state.tag[-1]

def firstCharBy3Tags(state):
    return 't35:'+state.tag[-3]+'_'+state.tag[-2]+'_'+state.word[-1][0]+'_'+state.tag[-1]

def tag0Tag1Size1(state):
    return 't36:'+state.tag[-2]+'_'+str(len(state.word[-2]))+'_'+state.tag[-1]

def tag1Tag2Size1(state):
    return 't37:'+state.tag[-3]+'_'+state.tag[-2]+'_'+str(len(state.word[-2]))

def tag0Tag1Tag2Size1(state):
    return 't38:'+state.tag[-3]+'_'+state.tag[-2]+'_'+str(len(state.word[-2]))+'_'+state.tag[-1]


#以下为word-classifier的特征

def start_end_punctuation(state,i,isNegative=False): # both start and end with punctuations
    if isNegative:
        if i > len(state.word)-4:
            return False
        return int(int(state.tag[i-1]=='w' and state.tag[i+2]=='w'))
    else:
        if i>len(state.word)-3:
            return False
        return int(int(state.tag[i-1]=='w' and state.tag[i+1]=='w'))


def count_p_bigger_20(state,i,noun2pattern,isNegative=False):
    if isNegative:
        count = len(noun2pattern.setdefault(state.word[i]+state.word[i+1],set()))
    else:
        count = len(noun2pattern.setdefault(state.word[i],set()))
    return int(count>19)

def count_p_between_10_20(state,i,noun2pattern,isNegative=False):
    if isNegative:
        count = len(noun2pattern.setdefault(state.word[i]+state.word[i+1],set()))
    else:
        count = len(noun2pattern.setdefault(state.word[i],set()))
    return int(count<20 and count>9)

def count_p_between_2_10(state,i,noun2pattern,isNegative=False):
    if isNegative:
        count = len(noun2pattern.setdefault(state.word[i]+state.word[i+1],set()))
    else:
        count = len(noun2pattern.setdefault(state.word[i],set()))
    return int(count<10 and count>=1)

def count_p_equal_1(state,i,noun2pattern,isNegative=False):
    if isNegative:
        count = len(noun2pattern.setdefault(state.word[i]+state.word[i+1], set()))
    else:
        count = len(noun2pattern.setdefault(state.word[i],set()))
    return int(count==0)

def freq_p_bigger_50(state,i,noun2pattern_freq,isNegative=False):
    if isNegative:
        freq = noun2pattern_freq.setdefault(state.word[i]+state.word[i+1],0)

    else:
        freq = noun2pattern_freq.setdefault(state.word[i],0)
    return int(freq>49)

def freq_p_between_20_50(state,i,noun2pattern_freq,isNegative=False):
    if isNegative:
        freq = noun2pattern_freq.setdefault(state.word[i]+state.word[i+1],0)

    else:
        freq = noun2pattern_freq.setdefault(state.word[i],0)
    return int(freq>19 and freq<50)

def freq_p_between_5_20(state,i,noun2pattern_freq,isNegative=False):
    if isNegative:
        freq = noun2pattern_freq.setdefault(state.word[i]+state.word[i+1],0)

    else:
        freq = noun2pattern_freq.setdefault(state.word[i],0)
    return int(freq>4 and freq<20)

def freq_p_smaller_5(state,i,noun2pattern_freq,isNegative=False):
    if isNegative:
        freq = noun2pattern_freq.setdefault(state.word[i]+state.word[i+1],0)

    else:
        freq = noun2pattern_freq.setdefault(state.word[i],0)
    return int(freq<5)

def PMI_1(word,word_freq,word_freq_sum):
    if len(word)<2:
        return 0
    f_1 = word_freq.setdefault(word[0],0)
    f_2 = word_freq.setdefault(word[1:],0)
    if f_1*f_2==0:
        return 0

    return word_freq.setdefault(word,0)*word_freq_sum/(f_1*f_2)

def PMI_n_1(word,word_freq,word_freq_sum):
    if len(word) < 2:
        return 0
    f_1 = word_freq.setdefault(word[:-1], 0)
    f_2 = word_freq.setdefault(word[-1], 0)
    if f_1 * f_2 == 0:
        return 0

    return word_freq.setdefault(word, 0) * word_freq_sum / (f_1 * f_2)

def PMI_2(word,word_freq,word_freq_sum):
    if len(word)<3:
        return 0

    f_1 = word_freq.setdefault(word[:2], 0)
    f_2 = word_freq.setdefault(word[2:], 0)
    if f_1 * f_2 == 0:
        return 0

    return word_freq.setdefault(word, 0) * word_freq_sum / (f_1 * f_2)

def PMI_n_2(word,word_freq,word_freq_sum):
    if len(word)<3:
        return 0

    f_1 = word_freq.setdefault(word[:-2], 0)
    f_2 = word_freq.setdefault(word[-2:], 0)
    if f_1 * f_2 == 0:
        return 0

    return word_freq.setdefault(word, 0) * word_freq_sum / (f_1 * f_2)





#以下为enhanced tagger新加的feature
def isNoun2Tag2Len2(state,W):
    return 'e1:'+str(int(state.word[-3] in W))+'_'+state.tag[-3]+'_'+str(len(state.word[-3]))

def isNoun1Tag1Len1(state,W):
    return 'e2:'+str(int(state.word[-2] in W))+'_'+state.tag[-2]+'_'+str(len(state.word[-2]))

def isNoun0Tag0Len0(state,W):
    return 'e3:'+str(int(state.word[-1] in W))+'_'+state.tag[-1]+'_'+str(len(state.word[-1]))

def istrippleTag1Len1_1(state,istrippleSet,W):
    istripple = str(int((state.word[-1] in W) and ((state.word[-3]+'_'+state.word[-1][0])in istrippleSet)))
    return 'e4:'+ istripple + state.tag[-2]+'_'+str(len(state.word[-1]))

def ispatternTag1Len1(state,ispatternSet):
    ispattern = str(int((state.word[-3] + '_' + state.word[-1][0]) in ispatternSet))
    return 'e5:'+ ispattern + state.tag[-2]+'_'+str(len(state.word[-1]))

# def istrippleTag1Len1_2(state,istrippleSet,W):
#     istripple = str(int((state.word[-1] in W) and ((state.word[-3]+'_'++state.word[-2]+'_'+state.word[-1][0])in istrippleSet)))
#     return 'e4:'+ istripple + state.tag[-2]+'_'+str(len(state.word[-1]))






