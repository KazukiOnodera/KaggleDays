#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 09:39:37 2018

@author: kazuki.onodera
"""

import pandas as pd
from time import time
import textblob
import gc
from multiprocessing import Pool
import utils
utils.start(__file__)
#==============================================================================


train = utils.load_train()
test  = utils.load_test()


question = pd.concat([train, test]).drop_duplicates('question_text')[['question_text']].reset_index(drop=True)
answer = pd.concat([train, test]).drop_duplicates('answer_text')[['answer_text']].reset_index(drop=True)


# =============================================================================
# def
# =============================================================================
def translate(x, lang='es'):
    """
    lang: en, es, ru, de, fr
    """
    try:
        return str(textblob.TextBlob(x).translate(to=lang))
    except:
        return x

def multi_q(args):
    ix, lang = args
    base = question.iloc[ix, 0]
    result = translate(base)
    
#    if ix%1000==0:
#        print(ix, base, result, round(utils.elapsed_minute(), 4))
    
    return result
    
def multi_a(args):
    ix, lang = args
    base = answer.iloc[ix, 0]
    result = translate(base)
    
    if ix%1000==0:
        print(ix, base, result, round(utils.elapsed_minute(), 4))
    
    return result
    
# =============================================================================
# translate
# =============================================================================

# est
#pool = Pool(16)
#st = time()
#args = zip(range(1000), ['es']*1000)
#result = pool.map(multi_q, args) # for test
#print(time()-st)
#pool.close()

# for Q
pool = Pool(16)
args = zip(range(len(question)), ['es']*range(len(question)))
result = pool.map(multi_q, range(len(question)))
question['q_es'] = result
utils.to_pickles(question, '../data/question_es', utils.SPLIT_SIZE)
pool.close()

pool = Pool(16)
args = zip(range(len(question)), ['de']*range(len(question)))
result = pool.map(multi_q, range(len(question)))
question['q_de'] = result
pool.close()

pool = Pool(16)
args = zip(range(len(question)), ['fr']*range(len(question)))
result = pool.map(multi_q, range(len(question)))
question['q_fr'] = result
pool.close()

# for A
pool = Pool(16)
args = zip(range(len(answer)), ['es']*range(len(answer)))
result = pool.map(multi_a, range(len(answer)))
answer['a_es'] = result
pool.close()

pool = Pool(16)
args = zip(range(len(answer)), ['de']*range(len(answer)))
result = pool.map(multi_a, range(len(answer)))
answer['a_de'] = result
pool.close()

pool = Pool(16)
args = zip(range(len(answer)), ['fr']*range(len(answer)))
result = pool.map(multi_a, range(len(answer)))
answer['a_fr'] = result
pool.close()


utils.to_pickles(question, '../data/question_tran', utils.SPLIT_SIZE)
utils.to_pickles(answer,   '../data/answer_tran', utils.SPLIT_SIZE)

#==============================================================================
utils.end(__file__)



#==============================================================================
utils.end(__file__)
