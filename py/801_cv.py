#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 14:15:51 2018

@author: kazuki.onodera
"""

import pandas as pd
import numpy as np
from glob import glob
import sys
sys.path.append('/home/kazuki_onodera/Python')
import xgbextension as ex
import utils
#utils.start(__file__)

FRAC = 0.2
SEED = 71

question_id = utils.load_train()[['question_id']]

#train_files = sorted(glob('../data/*_train'))
train_files = ['../data/101_train', '../data/102_train', 
               '../data/203_train', '../data/204_train', '../data/701_train']
print(train_files)

vec = pd.read_pickle('../data/train_vec.pkl')
# =============================================================================
# load
# =============================================================================
def subsample(folder):
    files = sorted(glob(folder+'/*'))
    df = pd.concat([pd.read_pickle(f).sample(frac=FRAC, random_state=SEED) for f in files])
    return df

X = pd.concat([question_id]+[utils.read_pickles(f) for f in train_files]+[vec], axis=1)
#X = pd.concat([question_id]+[subsample(f) for f in train_files], axis=1)
y = utils.read_pickles('../data/label')['answer_score'].map(np.log1p)

# =============================================================================
# xgb
# =============================================================================
params = {'max_depth':5, 
         'eta':0.1,
         'colsample_bytree':0.7,
         'silent':1,
         'eval_metric':'rmse',
         'objective':'reg:linear',
         'tree_method':'hist'}

yhat, imp, ret = ex.stacking(X, y, params, 9999, esr=50, by='question_id', seed=SEED)



