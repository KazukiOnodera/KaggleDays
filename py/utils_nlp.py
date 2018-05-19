#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 05:08:07 2018

@author: kazuki.onodera

https://github.com/ChenglongChen/Kaggle_HomeDepot
"""
import sys
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import pearsonr
from collections import Counter

try:
    import lzma
    import Levenshtein
except:
    pass
import numpy as np
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity

#from utils import np_utils
#sys.path.append("..")
#import config
from fastText import load_model

# =============================================================================
# global variables
# =============================================================================
MISSING_VALUE_NUMERIC = -1.

# =============================================================================
# np utils
# =============================================================================
def sigmoid(score):
    p = 1. / (1. + np.exp(-score))
    return p


def logit(p):
    return np.log(p/(1.-p))


def softmax(score):
    score = np.asarray(score, dtype=float)
    score = np.exp(score - np.max(score))
    score /= np.sum(score, axis=1)[:,np.newaxis]
    return score


def cast_proba_predict(proba):
    N = proba.shape[1]
    w = np.arange(1,N+1)
    pred = proba * w[np.newaxis,:]
    pred = np.sum(pred, axis=1)
    return pred


def one_hot_label(label, n_classes):
    num = label.shape[0]
    tmp = np.zeros((num, n_classes), dtype=int)
    tmp[np.arange(num),label.astype(int)] = 1
    return tmp


def majority_voting(x, weight=None):
    ## apply weight
    if weight is not None:
        assert len(weight) == len(x)
        x = np.repeat(x, weight)
    c = Counter(x)
    value, count = c.most_common()[0]
    return value


def voter(x, weight=None):
    idx = np.isfinite(x)
    if sum(idx) == 0:
        value = MISSING_VALUE_NUMERIC
    else:
        if weight is not None:
            value = majority_voting(x[idx], weight[idx])
        else:
            value = majority_voting(x[idx])
    return value


def array_majority_voting(X, weight=None):
    y = np.apply_along_axis(voter, axis=1, arr=X, weight=weight)
    return y


def mean(x):
    idx = np.isfinite(x)
    if sum(idx) == 0:
        value = float(MISSING_VALUE_NUMERIC) # cast it to float to accommodate the np.mean
    else:
        value = np.mean(x[idx]) # this is float!
    return value


def array_mean(X):
    y = np.apply_along_axis(mean, axis=1, arr=X)
    return y


def corr(x, y_train):
    if dim(x) == 1:
        corr = pearsonr(x.flatten(), y_train)[0]
        if str(corr) == "nan":
            corr = 0.
    else:
        corr = 1.
    return corr


def dim(x):
    d = 1 if len(x.shape) == 1 else x.shape[1]
    return d


def entropy(proba):
    entropy = -np.sum(proba*np.log(proba))
    return entropy


def try_divide(x, y, val=0.0):
    """try to divide two numbers"""
    if y != 0.0:
        val = float(x) / y
    return val

# =============================================================================
# dist utils
# =============================================================================
def edit_dist(str1, str2):
    try:
        # very fast
        # http://stackoverflow.com/questions/14260126/how-python-levenshtein-ratio-is-computed
        # d = Levenshtein.ratio(str1, str2)
        d = Levenshtein.distance(str1, str2)/float(max(len(str1),len(str2)))
    except:
        # https://docs.python.org/2/library/difflib.html
        d = 1. - SequenceMatcher(lambda x: x==" ", str1, str2).ratio()
    return d


def is_str_match(str1, str2, threshold=1.0):
    assert threshold >= 0.0 and threshold <= 1.0, "Wrong threshold."
    if float(threshold) == 1.0:
        return str1 == str2
    else:
        return (1. - edit_dist(str1, str2)) >= threshold


def longest_match_size(str1, str2):
    sq = SequenceMatcher(lambda x: x==" ", str1, str2)
    match = sq.find_longest_match(0, len(str1), 0, len(str2))
    return match.size


def longest_match_ratio(str1, str2):
    sq = SequenceMatcher(lambda x: x==" ", str1, str2)
    match = sq.find_longest_match(0, len(str1), 0, len(str2))
    return try_divide(match.size, min(len(str1), len(str2)))


def compression_dist(x, y, l_x=None, l_y=None):
    if x == y:
        return 0
    x_b = x.encode('utf-8')
    y_b = y.encode('utf-8')
    if l_x is None:
        l_x = len(lzma.compress(x_b))
        l_y = len(lzma.compress(y_b))
    l_xy = len(lzma.compress(x_b+y_b))
    l_yx = len(lzma.compress(y_b+x_b))
    dist = try_divide(min(l_xy,l_yx)-min(l_x,l_y), max(l_x,l_y))
    return dist


def cosine_sim(vec1, vec2):
    try:
        s = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    except:
        try:
            s = cosine_similarity(vec1, vec2)[0][0]
        except:
            s = MISSING_VALUE_NUMERIC
    return s


def vdiff(vec1, vec2):
    return vec1 - vec2


def rmse(vec1, vec2):
    vdiff = vec1 - vec2
    rmse = np.sqrt(np.mean(vdiff**2))
    return rmse


def KL(dist1, dist2):
    "Kullback-Leibler Divergence"
    return np.sum(dist1 * np.log(dist1/dist2), axis=1)


def jaccard_coef(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return try_divide(float(len(A.intersection(B))), len(A.union(B)))


def dice_dist(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return try_divide(2.*float(len(A.intersection(B))), (len(A) + len(B)))

# =============================================================================
# ngram utils
# =============================================================================
def unigrams(words):
    """
        Input: a list of words, e.g., ["I", "am", "Denny"]
        Output: a list of unigram
    """
    assert type(words) == list
    return words


def bigrams(words, join_string, skip=0):
    """
       Input: a list of words, e.g., ["I", "am", "Denny"]
       Output: a list of bigram, e.g., ["I_am", "am_Denny"]
       I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for k in range(1,skip+2):
                if i+k < L:
                    lst.append( join_string.join([words[i], words[i+k]]) )
    else:
        # set it as unigram
        lst = unigrams(words)
    return lst


def trigrams(words, join_string, skip=0):
    """
       Input: a list of words, e.g., ["I", "am", "Denny"]
       Output: a list of trigram, e.g., ["I_am_Denny"]
       I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in range(L-2):
            for k1 in range(1,skip+2):
                for k2 in range(1,skip+2):
                    if i+k1 < L and i+k1+k2 < L:
                        lst.append( join_string.join([words[i], words[i+k1], words[i+k1+k2]]) )
    else:
        # set it as bigram
        lst = bigrams(words, join_string, skip)
    return lst


def quadgrams(words, join_string):
    """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
        Output: a list of trigram, e.g., ["I_am_Denny_boy"]
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 3:
        lst = []
        for i in range(L-3):
            lst.append( join_string.join([words[i], words[i+1], words[i+2], words[i+3]]) )
    else:
        # set it as trigram
        lst = trigrams(words, join_string)
    return lst


def uniterms(words):
    return unigrams(words)


def biterms(words, join_string):
    """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
        Output: a list of biterm, e.g., ["I_am", "I_Denny", "I_boy", "am_Denny", "am_boy", "Denny_boy"]
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for j in range(i+1,L):
                lst.append( join_string.join([words[i], words[j]]) )
    else:
        # set it as uniterm
        lst = uniterms(words)
    return lst


def triterms(words, join_string):
    """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
        Output: a list of triterm, e.g., ["I_am_Denny", "I_am_boy", "I_Denny_boy", "am_Denny_boy"]
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in range(L-2):
            for j in range(i+1,L-1):
                for k in range(j+1,L):
                    lst.append( join_string.join([words[i], words[j], words[k]]) )
    else:
        # set it as biterm
        lst = biterms(words, join_string)
    return lst


def quadterms(words, join_string):
    """
        Input: a list of words, e.g., ["I", "am", "Denny", "boy", "ha"]
        Output: a list of fourterm, e.g., ["I_am_Denny_boy", "I_am_Denny_ha", "I_am_boy_ha", "I_Denny_boy_ha", "am_Denny_boy_ha"]
        I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 3:
        lst = []
        for i in range(L-3):
            for j in range(i+1,L-2):
                for k in range(j+1,L-1):
                    for l in range(k+1,L):
                        lst.append( join_string.join([words[i], words[j], words[k], words[l]]) )
    else:
        # set it as triterm
        lst = triterms(words, join_string)
    return lst


#_ngram_str_map = {
#    1: "Unigram",
#    2: "Bigram",
#    3: "Trigram",
#    4: "Fourgram",
#    5: "Fivegram",
#    12: "UBgram",
#    123: "UBTgram",
#}


def ngrams(words, ngram, join_string=" "):
    """wrapper for ngram"""
    if ngram == 1:
        return unigrams(words)
    elif ngram == 2:
        return bigrams(words, join_string)
    elif ngram == 3:
        return trigrams(words, join_string)
    elif ngram == 4:
        return quadgrams(words, join_string)
    elif ngram == 12:
        unigram = unigrams(words)
        bigram = [x for x in bigrams(words, join_string) if len(x.split(join_string)) == 2]
        return unigram + bigram
    elif ngram == 123:
        unigram = unigrams(words)
        bigram = [x for x in bigrams(words, join_string) if len(x.split(join_string)) == 2]
        trigram = [x for x in trigrams(words, join_string) if len(x.split(join_string)) == 3]
        return unigram + bigram + trigram


#_nterm_str_map = {
#    1: "Uniterm",
#    2: "Biterm",
#    3: "Triterm",
#    4: "Fourterm",
#    5: "Fiveterm",
#}


def _nterms(words, nterm, join_string=" "):
    """wrapper for nterm"""
    if nterm == 1:
        return uniterms(words)
    elif nterm == 2:
        return biterms(words, join_string)
    elif nterm == 3:
        return triterms(words, join_string)
    elif nterm == 4:
        return quadterms(words, join_string)


# =============================================================================
# 
# =============================================================================
def load_fasttext_wiki_en():
    return load_model('/home/kazuki_onodera/nlp_source/wiki.en.bin')

def sent2vec(sen, model, method='sum'):
    """
    sen: list
    model: w2v or fasttext model
    """
    
    M = []
    for w in sen:
        M.append(model.get_word_vector(w).astype(np.float32))
    M = np.array(M)
    
    if method=='sum':
        v = M.sum(axis=0)
    elif method=='mean':
        v = M.mean(axis=0)
    
    return v / np.sqrt((v**2).sum())

