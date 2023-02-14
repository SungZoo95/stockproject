import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import soynlp
from soynlp.noun import LRNounExtractor_v2
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 로드 및 확인 
target = pd.read_csv('filter.csv', encoding = 'UTF-8').drop(['Unnamed: 0'], axis=1)
#target.columns
#target.shape 

# train,test 분리 
d_train = target.sample(frac=0.70, random_state=2023)
#d_train.shape
#d_train
d_test = target.drop(d_train.index)
#d_test.shape

# 2. 데이터 전처리 
def clean_sents_df(company):
    target_company = d_train[d_train['기업명'] == company]
    df = target_company
    df['정제된 내용'] = df['제목+내용'].str.replace('\\[삭제된 게시물의 답글\\]',' ')
    df['정제된 내용'] = df['정제된 내용'].str.replace('제목+내용:',' ')
    df['정제된 내용'] = df['정제된 내용'].str.replace('[^가-힣]',' ').str.replace(' +',' ').str.strip()
    df = df[df['정제된 내용'] != '']
    df = df.reset_index(drop=True)
    return  df
def clean_test_df(company):
    target_company = d_test[d_test['기업명'] == company]
    df = target_company
    
    df['정제된 내용'] = df['제목+내용'].str.replace('\\[삭제된 게시물의 답글\\]',' ')
    df['정제된 내용'] = df['정제된 내용'].str.replace('제목+내용:',' ')
    df['정제된 내용'] = df['정제된 내용'].str.replace('[^가-힣]',' ').str.replace(' +',' ').str.strip()
    df = df[df['정제된 내용'] != '']
    df = df.reset_index(drop=True)
    return  df




clean_sents_df('포스코케미칼')
clean_sents_df('삼성전자')
