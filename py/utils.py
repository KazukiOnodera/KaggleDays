
"""

train:

test:


"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm
from itertools import combinations
from sklearn.model_selection import KFold
from time import time
from datetime import datetime
import gc

# =============================================================================
# global variables
# =============================================================================


SPLIT_SIZE = 20







# =============================================================================
# def
# =============================================================================
def start(fname):
    global st_time
    st_time = time()
    print("""
#==============================================================================
# START!!! {}    PID: {}    time: {}
#==============================================================================
""".format( fname, os.getpid(), datetime.today() ))
    
    send_line(f'START {fname}  time: {elapsed_minute():.2f}min')
    
    return

def end(fname):
    
    print("""
#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(fname))
    print('time: {:.2f}min'.format( elapsed_minute() ))
    
    send_line(f'FINISH {fname}  time: {elapsed_minute():.2f}min')
    
    return

def elapsed_minute():
    return (time() - st_time)/60


def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)

def to_pickles(df, path, split_size=3, inplace=True):
    """
    path = '../output/mydf'
    
    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'
    
    """
    if inplace==True:
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.reset_index(drop=True)
    gc.collect()
    mkdir_p(path)
    
    kf = KFold(n_splits=split_size)
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
        df.iloc[val_index].to_pickle(f'{path}/{i:03d}.p')
    return

def read_pickles(path, col=None):
    if col is None:
        df = pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*')))])
    else:
        df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
    return df

def to_feathers(df, path, split_size=3, inplace=True):
    """
    path = '../output/mydf'
    
    wirte '../output/mydf/0.f'
          '../output/mydf/1.f'
          '../output/mydf/2.f'
    
    """
    if inplace==True:
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.reset_index(drop=True)
    gc.collect()
    mkdir_p(path)
    
    kf = KFold(n_splits=split_size)
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
        df.iloc[val_index].to_feather(f'{path}/{i:03d}.f')
    return

def read_feathers(path, col=None):
    if col is None:
        df = pd.concat([pd.read_feather(f) for f in tqdm(sorted(glob(path+'/*')))])
    else:
        df = pd.concat([pd.read_feather(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
    return df

def reduce_memory(df, ix_start=0):
    df.fillna(-1, inplace=True)
    if df.shape[0]>9999:
        df_ = df.sample(9999, random_state=71)
    else:
        df_ = df
    ## int
    col_int8 = []
    col_int16 = []
    col_int32 = []
#    for c in tqdm(df.columns[ix_start:], miniters=20):
    for c in df.columns[ix_start:]:
        if df[c].dtype=='O':
            continue
        if (df_[c] == df_[c].astype(np.int8)).all():
            col_int8.append(c)
        elif (df_[c] == df_[c].astype(np.int16)).all():
            col_int16.append(c)
        elif (df_[c] == df_[c].astype(np.int32)).all():
            col_int32.append(c)
    
    df[col_int8]  = df[col_int8].astype(np.int8)
    df[col_int16] = df[col_int16].astype(np.int16)
    df[col_int32] = df[col_int32].astype(np.int32)
    
    ## float
    col = [c for c in df.dtypes[df.dtypes==np.float64].index if '_id' not in c]
    df[col] = df[col].astype(np.float32)

    gc.collect()

def load_train():
    return read_pickles('../data/train')

def load_test():
    return read_pickles('../data/test')

# =============================================================================
# NLP
# =============================================================================




# =============================================================================
# other API
# =============================================================================
def submit(file_path, comment='from API'):
    os.system('kaggle competitions submit -c avito-demand-prediction -f {} -m "{}"'.format(file_path, comment))

import requests
def send_line(message):
    
    line_notify_token = '9oQ6pIvGUvh8zrhGJhp63cZfKxAgHa2BsdE5UBugsVR'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}  # 発行したトークン
    requests.post(line_notify_api, data=payload, headers=headers)


#import re
#
#url = 'https://translate.google.co.jp/?client=ubuntu&channel=fs&oe=utf-8&hl=ja&um=1&ie=UTF-8&hl=en&client=tw-ob#ru/en/'
##headers = {"User-Agent": "Chrome/58.0.3029.100"}
#pattern = "TRANSLATED_TEXT=\'(.*?)\'"
#
#def translate(text):
#    params = {'q': text}
#    headers = {"User-Agent": f"Chrome/58.0.3029.{np.random.randint(50,100)}"}
#    r = requests.get(url, headers=headers, params=params)
#    result = re.search(pattern, r.text).group(1)
#    return result
#
#translate('Кроссовки р.25 Котофей натуральная кожа, по стельке 15.5, состояние идеальное.')

#from googletrans import Translator
#translator = Translator()
#
#def translate(text):
#    try:
#        return translator.translate(text, src='ru', dest='en').text
#    except:
#        return "UNABLE TO TRANSLATE"


