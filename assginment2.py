# import pandas as pd
import re
import requests
import collections
from collections import Counter
from functools import reduce
from operator import mul, add
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from matplotlib.pyplot import  yscale,xscale,title,plot,show
import matplotlib.pyplot as plt
import jieba


def cut_word(string):
    l = jieba.lcut(string)
    return l


def tokenize(string):
    # 正则表达式选择 中文和数字部分, 原本有5.3亿字符，去掉英文子母后剩下3.7亿
    return ''.join(re.findall('[\u4e00-\u9fa5|\d]+', string))


# Unigram P(W0W1W2Wn)=P(W0)P(W1)P(w2)
def get_char_probability(char):
    all_occurences = sum(all_char_counts.values())
    return all_char_counts[char]/all_occurences

""" 阈值，当大于threshold,根据Katz的改进，不用平滑。1和threshold之间平滑，同时给为看到的分配一直值"""
def get_probability_from_counts(count,generate_nr_list):
    all_occurences = sum(count.values())
    un_seen = min(count.values())
    l =generate_nr_list(count)
    def get_prob(item):
        if count[item]>threshold:
            return count[item]/all_occurences
        elif count[item]>=1:
            return (count[item]+1)*l[count[item]]/(l[count[item-1]]*all_occurences)
        else : return un_seen/all_occurences
    return get_prob

# 最快的得到Nr的方程，index是r, 算法复杂度O（n）
def fast_nr_list(count):
    l=[0 for i in range(count.most_common()[0][1])]
    for (w,c) in count.items():
        l[c-1] += 1
    return l

# 和上一个方程类似，但初始化时列表，another to build a dict store :{ key=r,value=Nr}
def build_r_nr_dict(count):
    l = collections.defaultdict(list)
    nr_list=[]
    for (w,c) in count.items():
        l[c].append(1)
    return [sum(l[c]) for c in range(1,count.most_common()[0][1])]

def get_string_probability(string):
    return reduce(mul,[get_char_prob(c) for c in string])

def get_probability_peformance(language_model_func,pairs):
    for (p1,p2) in pairs:
        print('\t {} with probability {}'.format(p1,language_model_func(cut_word(tokenize(p1)))))
        print('\t {} with probability {}'.format(p2,language_model_func(cut_word(tokenize(p2)))))

# 2-gram
def get_2_gram_prob(word, prev):
    if get_pair_prob(prev+word) > 0:
        return get_pair_prob(prev+word)/get_char_prob(prev)
    else:
        return get_char_probability(word)

def get_2gram_string_prob(string):
    probabilites = []
    for i,c in enumerate (string):
        if i==0:
            prev = '<s>'
        else:
            prev = string[i-1]
        probabilites.append(get_2_gram_prob(c,prev))
    return reduce(mul, probabilites)

def plot():
    yscale('log');xscale('log');title('Frequency of frequencies ')
    plt.xlabel('log r,  which r stands for frequency which give number of individuals have been observed for species x  ')
    plt.ylabel('log Nr ,the frequency of frequencies ')
    plot(range(1,all_char_counts.most_common()[0][1]+1),fast_nr_list(all_char_counts),'ro')
    plot(range(1,two_gram_counts.most_common()[0][1]+1),fast_nr_list(two_gram_counts),'bs')
    #
    #
    # # plot([c for (w,c) in two_gram_counts.most_common()])
    # # plot([M/i for i in range (1,len(two_gram_counts)+1)])
    # # plot([c for (w,c) in all_char_counts.most_common()])
    # # plot([N/i for i in range (1,len(all_char_counts)+1)])
    show()


name_file = "80k_articles.txt"
all_content = open(name_file).read()
ALL_CHARACTER = tokenize(all_content)
# 分词
seg_dic = jieba.lcut(ALL_CHARACTER, cut_all=True)
all_char_counts = Counter(seg_dic)

pair = """前天晚上吃晚饭的时候
前天晚上吃早饭的时候""".split('\n')

pair2 = """正是一个好看的小猫
真是一个好看的小猫""".split('\n')

pair3 = """我无言以对，简直
我简直无言以对""".split('\n')

pairs = [pair, pair2, pair3]

# threshold for katz，这里取9
threshold = 9
get_char_prob = get_probability_from_counts(all_char_counts, fast_nr_list)
print(get_probability_peformance(get_string_probability, pairs))

two_gram_counts = Counter([tuple(seg_dic[i:i+2]) for i in range(len(seg_dic)-1)])
get_pair_prob = get_probability_from_counts(two_gram_counts, fast_nr_list)
print(get_probability_peformance(get_2gram_string_prob, pairs))



