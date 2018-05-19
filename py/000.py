#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 09:27:19 2018

@author: kazuki.onodera
"""
from os import system
import pandas as pd
import utils
utils.start(__file__)
#==============================================================================

system('rm -rf ../data')
system('mkdir ../data')

train = pd.read_table('../input/train.csv.zip', delimiter='\t')
test  = pd.read_table('../input/test.csv.zip', delimiter='\t')

train['question_utc'] = pd.to_datetime(train['question_utc'])
train['answer_utc'] = pd.to_datetime(train['answer_utc'])

test['question_utc'] = pd.to_datetime(test['question_utc'])
test['answer_utc'] = pd.to_datetime(test['answer_utc'])


utils.to_pickles(train, '../data/train', utils.SPLIT_SIZE)
utils.to_pickles(test, '../data/test',   utils.SPLIT_SIZE)

#==============================================================================
utils.end(__file__)
