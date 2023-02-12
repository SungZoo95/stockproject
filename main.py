# <pythonfile>.ipynb notebook
# 프로그램 작성시에는 <pythonfile>.py
# python3 <pythonfile>.py
# streamlit run <streamlitapp>.py



import streamlit as st
import pandas as pd
import numpy as np
# pip install pandas
# conda install pandas

def text():
    #Mark Down
    st.markdown('------')
    st.markdown('네이버 증권에서 제공하는 종목 토론실에서 인기 검색종목 30개 기업을 뽑았습니다.')
    st.markdown('뽑은 30개 기업들에 대한 종목 토론실을 크롤링하여 댓글들의 데이터를 뽑았습니다.')
    st.markdown("뽑은 데이터들로 토큰화 작업을 진행 후 감정분석 작업을 실시하였습니다.")
    st.markdown("- 긍정 : 1")
    st.markdown("- 중립 : 0")
    st.markdown("- 부정 : -1")

def dataframe1():
    df = pd.DataFrame(
    np.random.randn(50, 20),
    columns=('col %d' % i for i in range(20)))
    st.dataframe(df) # Same as st.write(df)

def temp_map():
    # 온도 출력
    st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
    # 위도 경도에 맞는 지도를 출력
    df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
    st.map(df)


def main():
    st.title("자연어 처리/추천시스템 프로젝트:blue_book:")

    #st.sidebar.write('''
    ## lab1
    ## lab2
    #- lab3
    #- lab4
    #''')

    code = '''이번 프로젝트를 통해 네이버 종목토론실의 댓글을 통해 감정 분석을 사용해 \n다음날의 주가 상승률을 예측하고 크게는 추천해보는 프로젝트를 진행했습니다.'''
    st.code(code, language='python')

    text()

    if st.checkbox("show dataframe"):
        dataframe1()

    temp_map()

if __name__ == "__main__":
    main()
