#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 13:29:28 2018

@author: kazuki.onodera
"""

import pandas as pd
import numpy as np
import gc
from multiprocessing import Pool
import utils
utils.start(__file__)
#==============================================================================


train = utils.load_train()[['question_id']]
test  = utils.load_test()[['question_id']]

train = pd.concat([train, utils.read_pickles('../data/101_train')], axis=1)
test = pd.concat([test, utils.read_pickles('../data/101_test')], axis=1)

gc.collect()

USECOLS = ['a_dow', 'a_hour', 'timediff_a-q', 'a_len', 'a_count_words', 'a_count_unq_words']

def nunique(x):
    return len(set(x))

def make_features(p):
    
    if p==0:
        df=train
        name='train'
    else:
        df=test
        name='test'
        
    init_col = df.columns.tolist()
    
    gr = df.groupby('question_id')
    
    for c in USECOLS:
        print(name, c)
        df[f'{c}_min'] = gr[c].transform(np.min)
        df[f'{c}_max'] = gr[c].transform(np.max)
        df[f'{c}_max-min'] = df[f'{c}_max'] - df[f'{c}_min']
        df[f'{c}_mean'] = gr[c].transform(np.mean)
        df[f'{c}_std'] = gr[c].transform(np.std)
        df[f'{c}_nunique'] = gr[c].transform(nunique)
    
    df.drop(init_col, axis=1, inplace=True)
    utils.to_pickles(df, f'../data/102_{name}', utils.SPLIT_SIZE)

# =============================================================================
# 
# =============================================================================

pool = Pool(2)
pool.map(make_features, [0, 1])
pool.close()



#==============================================================================
utils.end(__file__)

