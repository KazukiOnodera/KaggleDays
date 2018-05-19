#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 09:39:37 2018

@author: kazuki.onodera
"""

import pandas as pd
import numpy as np
from time import time
from multiprocessing import Pool
import utils_nlp
import utils
utils.start(__file__)
#==============================================================================

train = utils.read_pickles('../data/train_stop')
test  = utils.read_pickles('../data/test_stop')


wiki_en = utils_nlp.load_fasttext_wiki_en()
valid_words = set(wiki_en.get_words())
DIM = wiki_en.get_dimension()


def get_vector_with_words_sum(s):
    sen = [w for w in s if w in valid_words]
    if len(sen)==0:
        return np.zeros(DIM)-1
    return utils_nlp.sent2vec(sen, wiki_en, method='sum')

def get_vector_with_words_mean(s):
    sen = [w for w in s if w in valid_words]
    if len(sen)==0:
        return np.zeros(DIM)-1
    return utils_nlp.sent2vec(sen, wiki_en, method='mean')

st_time = time()
def make_features(p):
    if p==0:
        df=train
        name='train'
    else:
        df=test
        name='test'
    
    # get vec
    print(name, 'sum', round(st_time - time(), 4))
    question_vec_sum = pd.DataFrame(list(df['q_stop'].map(get_vector_with_words_sum))).add_prefix('q_stop_vec_sum_')
    answer_vec_sum   = pd.DataFrame(list(df['a_stop'].map(get_vector_with_words_sum))).add_prefix('a_stop_vec_sum_')
    
    print(name, 'mean', round(st_time - time(), 4))
    question_vec_mean = pd.DataFrame(list(df['q_stop'].map(get_vector_with_words_mean))).add_prefix('q_stop_vec_mean_')
    answer_vec_mean   = pd.DataFrame(list(df['a_stop'].map(get_vector_with_words_mean))).add_prefix('a_stop_vec_mean_')
    
    result = pd.concat([question_vec_sum, answer_vec_sum, question_vec_mean, answer_vec_mean], axis=1)
    
    utils.to_pickles(result, f'../data/202_{name}', utils.SPLIT_SIZE)
    
# =============================================================================
# 
# =============================================================================


pool = Pool(2)
pool.map(make_features, [0, 1])
pool.close()


#==============================================================================
utils.end(__file__)
