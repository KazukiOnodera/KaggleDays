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
#import utils
#utils.start(__file__)
#==============================================================================


train = pd.read_table('../input/train.csv.zip', delimiter='\t', usecols=['question_text', 'answer_text'])
test  = pd.read_table('../input/test.csv.zip', delimiter='\t', usecols=['question_text', 'answer_text'])


question = pd.concat([train, test]).drop_duplicates('question_text')[['question_text']].reset_index(drop=True)
answer = pd.concat([train, test]).drop_duplicates('answer_text')[['answer_text']].reset_index(drop=True)

st_time = time()
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
    
    if ix%1000==0:
        print(ix, base, result, round(st_time - time(), 4))
    
    return result
    
def multi_a(args):
    ix, lang = args
    base = answer.iloc[ix, 0]
    result = translate(base)
    
    if ix%1000==0:
        print(ix, base, result, round(st_time - time(), 4))
    
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
args = zip(range(len(question)), ['es']*len(question))
result = pool.map(multi_q, args)
question['q_es'] = result
pool.close()

#pool = Pool(16)
#args = zip(range(len(question)), ['de']*len(question))
#result = pool.map(multi_q, args)
#question['q_de'] = result
#pool.close()
#
#pool = Pool(16)
#args = zip(range(len(question)), ['fr']*len(question))
#result = pool.map(multi_q, args)
#question['q_fr'] = result
#pool.close()

# for A
pool = Pool(16)
args = zip(range(len(answer)), ['es']*len(answer))
result = pool.map(multi_a, args)
answer['a_es'] = result
pool.close()

#pool = Pool(16)
#args = zip(range(len(answer)), ['de']*len(answer))
#result = pool.map(multi_a, args)
#answer['a_de'] = result
#pool.close()
#
#pool = Pool(16)
#args = zip(range(len(answer)), ['fr']*len(answer))
#result = pool.map(multi_a, args)
#answer['a_fr'] = result
#pool.close()


question.to_csv('../data/question_tran.csv.gz', index=False, compression='gzip')
answer.to_csv('../data/answer_tran.csv.gz', index=False, compression='gzip')

