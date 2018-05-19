#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 14:18:40 2018

@author: kazuki.onodera
"""

import utils
utils.start(__file__)

y = utils.load_train[['answer_score']]

utils.to_pickles(y, f'../data/label', utils.SPLIT_SIZE)



#==============================================================================
utils.end(__file__)
