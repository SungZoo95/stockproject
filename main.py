# <pythonfile>.ipynb notebook
# 프로그램 작성시에는 <pythonfile>.py
# python3 <pythonfile>.py
# streamlit run <streamlitapp>.py
# pip install pandas
# conda install pandas

import streamlit as st
import pandas as pd
import numpy as np
import datetime
from PIL import Image
import matplotlib.pyplot as plt


def text():
    #Mark Down
    st.markdown('네이버 증권에서 제공하는 종목 토론실에서 인기 검색종목 30개 기업을 뽑았습니다.')
    st.markdown('뽑은 30개 기업들에 대한 종목 토론실을 크롤링하여 댓글들의 데이터를 뽑았습니다.')
    st.markdown("뽑은 데이터들로 토큰화 작업을 진행 후 감성분석 작업을 실시하였습니다.")
    st.markdown("- 긍정 : 1:grinning:")
    st.markdown("- 중립 : 0:zipper_mouth_face:")
    st.markdown("- 부정 : -1:angry:")
    st.markdown("이모티콘은 댓글 성향에 따른 감성상태를 나타냅니다.")
    
    
def dataframe1():
    df = pd.read_csv('name_code_0206.csv', dtype=str)
    st.dataframe(df) # Same as st.write(df)


def stock_date_input():
    df_1 = pd.read_csv('name_code_0206.csv', dtype=str)
    df_2 = pd.read_csv('naver60pages.csv')
    df_2.drop(['Unnamed: 0'],axis=1,inplace=True)
    
    da = st.date_input(
        "날짜를 선택하세요",
        datetime.date(2023,2,8), max_value=datetime.date(2023,2,10), min_value=datetime.date(2023,1,29))
    st.write('선택한 날짜는:', da)
    
    stock = st.selectbox(
        '기업을 선택하세요',
        (df_1["종목명"]))
    st.write('선택한 기업은:',stock)
    
    df_3 = df_2[(df_2["기업명"]==f'{stock}') & (df_2["날짜"]==f'{da}')]
    shape = df_3.shape[0]
    st.dataframe(df_3)
    st.write(f"총 {shape}개의 행이 출력되었습니다")

def dataframe2():
    df = pd.read_csv('top30stock.csv')
    df.drop(["Unnamed: 0"],axis=1, inplace=True)
    st.dataframe(df)

def dataframe2_add():
    key = "unique_key"
    df_1 = pd.read_csv('name_code_0206.csv', dtype=str)
    stock = st.selectbox(
        '기업을 선택하세요',
        (df_1["종목명"]),key=key)
    
    df_2 = pd.read_csv('top30stock.csv')
    df_2.drop(["Unnamed: 0"],axis=1, inplace=True)
    check = st.multiselect(
        '원하시는 정보를 선택하세요',(df_2.columns[:]))
    st.dataframe(df_2[df_2["종목명"]==stock][check])
    df_3 = df_2[df_2["종목명"]==stock][check].shape[0]
    st.write(f"총 {df_3}개의 행이 출력되었습니다")
   
def makegraph():
    df_1 = pd.read_csv('name_code_0206.csv')
    df_2 = pd.read_csv('top30stock.csv')
    key = "key_1"
    st.subheader('기업별 종가그래프')
    option = st.selectbox(
        '기업을 선택하세요', (df_1["종목명"]),key=key)

    stock_data = df_2[(df_2['종목명']==option)][["날짜","종가"]]
    stock_data["날짜"] = pd.to_datetime(stock_data["날짜"])
    stock_data["종가"] = stock_data["종가"].str.replace(',',"")
    stock_data["종가"] = stock_data["종가"].astype(np.int64)
    x = range(len(stock_data["날짜"]))
    fig = plt.figure()
    plt.bar(x=x, height=stock_data["종가"], width=0.5, color='gray')
    plt.xticks(x, stock_data["날짜"].dt.strftime("%Y-%m-%d"),rotation=90)
    plt.ylim(min(stock_data["종가"]-10000), max(stock_data["종가"]+3000))
    st.pyplot(fig)
   
  
def download(file):
    with open(file, 'r') as file:
        csv_data = file.read()
    st.download_button(
        label="Download File",
        data=csv_data,
        file_name='name_code_0206.csv',
        mime='text/csv',
    )

#def image():
#    
#    image = Image.open('heechan.jpg')
#    st.image(image, caption='Sunrise by the mountains')


def main():
    st.title("자연어 처리/추천시스템 프로젝트:blue_book:")

    #st.sidebar.write('''
    ## lab1
    ## lab2
    #- lab3
    #- lab4
    #''')

    code = '''이번 프로젝트를 통해 네이버 종목토론실의 댓글을 통해 감성 분석을 사용해 \n다음날의 주가 상승률을 예측하고 크게는 추천해보는 프로젝트를 진행했습니다.'''
    st.code(code, language='python')
    st.markdown('------')
    
    download('name_code_0206.csv')
    st.markdown('30개 기업코드 다운로드')
    
    if st.checkbox("30개 기업확인"):
        dataframe1()
    
    text()
    st.markdown('------') 
    
    st.header('날짜/기업별 댓글 조회')
    stock_date_input()
    st.markdown('------')

    st.header('기업정보')
    download('top30stock.csv')
    st.markdown('삼성 주식정보 다운로드')
    if st.checkbox("기업정보"):
        dataframe2()
    dataframe2_add()   
    st.markdown('-----')
    makegraph()
   
    
if __name__ == "__main__":
    main()
