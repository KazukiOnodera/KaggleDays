#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 11:19:50 2018

@author: kazuki.onodera
"""

import pandas as pd
from nltk.corpus import stopwords
import utils
utils.start(__file__)
#==============================================================================

train = utils.load_train()
test  = utils.load_test()

stopwords = set(stopwords.words('english'))

def remve_stop(s):
    s = [w for w in s.split() if s not in stopwords]
    return s

def make_features(df):
    init_col = df.columns.tolist()
    
    # length features
    df['q_stop'] = df['question_text'].map(remve_stop)
    df['a_stop'] = df['answer_text'].map(remve_stop)
    
    df.drop(init_col, axis=1, inplace=True)

# =============================================================================
# main
# =============================================================================

make_features(train)
make_features(test)

utils.to_pickles(train, '../data/train_stop', utils.SPLIT_SIZE)
utils.to_pickles(test, '../data/test_stop',   utils.SPLIT_SIZE)


#==============================================================================
utils.end(__file__)

