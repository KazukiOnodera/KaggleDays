#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:10:49 2018

@author: kazuki.onodera
"""

import utils


train = utils.read_pickles('../data/201_train')
test  = utils.read_pickles('../data/201_test')

train.iloc[:600].to_pickle('../data/train_vec.pkl')
test.iloc[:600].to_pickle('../data/test_vec.pkl')



