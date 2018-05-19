#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:21:44 2018

@author: kazuki.onodera

from TeraFlops
"""

#from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from multiprocessing import Pool
import utils
#utils.start(__file__)
#==============================================================================

train = utils.load_train()
test  = utils.load_test()

def splittext(x):
   return x.replace('.',' ').replace(',',' ').replace(':',' ').replace(';',' ').replace('#',' ').replace('!',' ').split(' ')
   #return x.split(' ')
   
def make_features(p):
    if p==0:
        df=train
        name='train'
    else:
        df=test
        name='test'
    init_col = df.columns.tolist()
    print(init_col)
    
    df['qlenchar'] = df.question_text.apply(len)
    df['qlenword'] = df.question_text.apply(lambda x:len(splittext(x)))
    df['alenchar'] = df.answer_text.apply(len)
    df['alenword'] = df.answer_text.apply(lambda x:len(splittext(x)))
    
    df['difflenchar'] = df.qlenchar - df.alenchar
    df['difflenword'] = df.qlenword - df.alenword
    
    df['divlenchar'] = df.qlenchar / df.alenchar
    df['divlenword'] = df.qlenword / df.alenword
    
    df['idivlenchar'] = df.alenchar / df.qlenchar
    df['idivlenword'] = df.alenword / df.qlenword
    
#    df['subreddit_le'] = LabelEncoder().fit_transform(df.subreddit)
#    df['qid'] = LabelEncoder().fit_transform(df.question_id)
    df = pd.get_dummies(df, columns=['subreddit'])
    init_col.remove('subreddit')
    
    df['qdt_dow'] = pd.to_datetime(df.question_utc,origin='unix',unit='s').dt.dayofweek
    df['qdt_hour'] = pd.to_datetime(df.question_utc,origin='unix',unit='s').dt.hour
    
    df['adt_dow'] = pd.to_datetime(df.answer_utc,origin='unix',unit='s').dt.dayofweek
    df['adt_hour'] = pd.to_datetime(df.answer_utc,origin='unix',unit='s').dt.hour
    
#    df['question_score_l1p'] = np.log1p(df.question_score)
#    df['answer_score_l1p'] = np.log1p(df.answer_score)
    
    df['qboldwords'] = df.question_text.apply(lambda x:np.sum(x.isupper() for x in splittext(x) if len(x)>1))
    df['aboldwords'] = df.answer_text.apply(lambda x:np.sum(x.isupper() for x in splittext(x) if len(x)>1))
    
    
    df.drop(init_col, axis=1, inplace=True)
    print(name, df.columns.tolist())
    utils.to_pickles(df, f'../data/701_{name}', utils.SPLIT_SIZE)

# =============================================================================
# 
# =============================================================================


pool = Pool(2)
pool.map(make_features, [0, 1])
pool.close()


#==============================================================================
utils.end(__file__)


