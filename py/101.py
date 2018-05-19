#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 09:26:09 2018

@author: kazuki.onodera

basic features
"""

import pandas as pd
import utils
utils.start(__file__)
#==============================================================================

train = utils.load_train()
test  = utils.load_test()

def count_words(s):
    return len( s.split() )

def count_unq_words(s):
    return len( set(s.split()) )

def make_features(df):
    # datetime features
    df['q_dow']  = df['question_utc'].dt.dayofweek
    df['q_hour'] = df['question_utc'].dt.hour
    df['a_dow']  = df['answer_utc'].dt.dayofweek
    df['a_hour'] = df['answer_utc'].dt.hour
    
    # length features
    df['q_len'] = df['question_text'].map(len)
    df['a_len'] = df['answer_text'].map(len)
    df['q_count_words'] = df['question_text'].map(count_words)
    df['a_count_words'] = df['answer_text'].map(count_words)
    df['q_count_unq_words'] = df['question_text'].map(count_unq_words)
    df['a_count_unq_words'] = df['answer_text'].map(count_unq_words)

# =============================================================================
# main
# =============================================================================

make_features(train)
make_features(test)

utils.to_pickles(train, '../data/101_train', utils.SPLIT_SIZE)
utils.to_pickles(test, '../data/101_test',   utils.SPLIT_SIZE)

#==============================================================================
utils.end(__file__)
