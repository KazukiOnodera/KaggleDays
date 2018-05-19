#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:14:12 2018

@author: kazuki.onodera
"""

import pandas as pd
from multiprocessing import Pool
import utils_nlp
import utils
utils.start(__file__)
#==============================================================================

train = utils.read_pickles('../data/202_train')
test  = utils.read_pickles('../data/202_test')

def get_sim(s):
    vec1, vec2, vec1_, vec2_ = s.iloc[0:300], s.iloc[300:600], s.iloc[600:900], s.iloc[900:1200]
    
    cosine_sim  = utils_nlp.cosine_sim(vec1, vec2)
    cosine_sim_ = utils_nlp.cosine_sim(vec1_, vec2_)
    
    return cosine_sim, cosine_sim_

def make_features(p):
    if p==0:
        df=train
        name='train'
    else:
        df=test
        name='test'
    
    init_col = df.columns.tolist()
    
    vec_df = df.apply(get_sim, axis=1)
    df['cosine_sim'] = vec_df.apply(lambda x: x[0])
    df['cosine_sim_mean'] = vec_df.apply(lambda x: x[1])
    
    df.drop(init_col, axis=1, inplace=True)
    
    utils.to_pickles(df, f'../data/204_{name}', utils.SPLIT_SIZE)

# =============================================================================
# 
# =============================================================================


pool = Pool(2)
pool.map(make_features, [0, 1])
pool.close()


#==============================================================================
utils.end(__file__)

