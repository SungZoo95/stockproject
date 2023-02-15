import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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

# 안나오는 함수?
clean_sents_df('포스코케미칼')
clean_sents_df('삼성전자')

#posco_df = clean_sents_df('포스코케미칼')
#samsung_df = clean_sents_df('삼성전자')
#
#st.write("POSCO dataframe:")
#st.write(posco_df)
#
#st.write("Samsung dataframe:")
#st.write(samsung_df)

company_list = list(d_train["기업명"])
company_list = list(set(company_list))
#company_list

len(company_list)

# Labeling을 위한 토큰화 작업 
from konlpy.tag import Mecab
def corpus_save(company):
    df = clean_sents_df(company)
    df['정제된 내용 길이'] = [len(str(i)) for i in df['정제된 내용']]

    tp = [str(i) for i in list(df['정제된 내용'])]
    save = '\n'.join(tp)
    f = open("vocab.txt", 'a',encoding='utf8')
    f.write(save)
    f.close()
    
def corpus_init():
    #company_set.append('sampro') # 현재 주피터노트북에서 제외
    f = open("vocab.txt", 'w',encoding='utf8')
    f.write('')
    f.close()
    for company in company_list:
        corpus_save(company)

def return_tokenizer():
    corpus = DoublespaceLineCorpus("/vocab.txt",iter_sent=True)
    noun_extractor = LRNounExtractor_v2(verbose=True)
    nouns = noun_extractor.train_extract(corpus)
    scores = {word:score.score for word, score in nouns.items()}
    tokenizer = LTokenizer(scores=scores)
    return tokenizer

corpus_init()
tokenizer = return_tokenizer()

def labeling(target_df):
    f = open("neg_pol_word.txt", 'r',encoding='utf8')
    words = f.readlines()
    f.close()
    fear_words_set = {word.strip('\n') for word in words}

    f = open("pos_pol_word.txt", 'r',encoding='utf8')
    words = f.readlines()
    f.close()
    greed_words_set = {word.strip('\n') for word in words}
    
    label_score = []
    for token_list in target_df['토큰화']:
        sent_score = 0
        for token in token_list:
            if token in fear_words_set:
                sent_score -= 1
            elif token in greed_words_set:
                sent_score += 1

        if sent_score < 0:
            label_score.append(-1)
        elif sent_score > 0:
            label_score.append(1)
        else:
            label_score.append(0)
            
    target_df['label'] = label_score
    return target_df


#train 데이터 구조 형성 
def setting_train_data():
    company = company_list

    train_data = pd.DataFrame(columns=['날짜', '기업명', '제목+내용', '조회수', '공감수','비공감수','페이지', '전날_대비_상승_하락',
                                       '정제된 내용', '토큰화', 'label'])
    for idx in range(len(company)):
        target_df = clean_sents_df(company[idx])
        target_df['토큰화'] = [tokenizer(str(i)) for i in target_df['정제된 내용']]

        label_df = labeling(target_df)
        train_data = train_data.append(label_df)
    
    return train_data
train_data = setting_train_data()
train_data
train_data.reset_index(inplace=True)
train_data.drop('index',axis=1,inplace=True)
train_data
