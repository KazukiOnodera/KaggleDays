#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 15:28:58 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib_venn import venn2

def df_info(target_df):
    
    print(f'Shape: {target_df.shape}')
    
    df = target_df.dtypes.to_frame()
    df.columns = ['DataType']
    df['#Nulls'] = target_df.isnull().sum()
    df['#Uniques'] = target_df.nunique()
    
    return df

def top_categories(df, category_feature, top=30):
    return df[category_feature].value_counts()[:top].index

def count_categories(df, category_features, top=30, sort='freq'):
    
    for c in category_features:
        target_value = df[c].value_counts()[:top].index
        if sort=='freq':
            order = target_value
        elif sort=='alphabetic':
            order = df[c].value_counts()[:top].sort_index().index
        sns.countplot(x=c, data=df[df[c].isin(order)], order=order)
        plt.title(f'{c} TOP{top}', size=30)
        plt.xticks(rotation=90)
        plt.show()
        
    return

def venn_diagram(train, test, category_features):
    """
    category_features: max==6
    """
    n = int(np.ceil(len(category_features)/2))
    plt.figure(figsize=(18,13))
    
    for i,c in enumerate(category_features):
        plt.subplot(int(f'{n}2{i+1}'))
        venn2([set(train[c].unique()), set(test[c].unique())], set_labels = ('Train set', 'Test set') )
        plt.title(f'{c}', fontsize=18)
    plt.show()
    
    return
