import streamlit as st
import pandas as pd
import numpy as np
import datetime
from PIL import Image
import matplotlib.pyplot as plt


def makemodel():
    df_1 = pd.read_csv("name_code_0206.csv", dtype=str)
    # 날짜 달력
    da = st.date_input(
        "날짜를 선택하세요",
        datetime.date(2023,2,9), max_value=datetime.date(2023,2,10), min_value=datetime.date(2023,2,1))
    # 기업 선택창
    stock = st.selectbox(
        '기업을 선택하세요',
        (df_1["종목명"]))
    df_2 = pd.read_csv("konlpydata.csv").drop(["Unnamed: 0.1", "Unnamed: 0"],axis=1)
    df_2 = df_2[(df_2["날짜"]==f"{da}") & (df_2["기업명"]==f"{stock}")]
    if df_2.empty:
        st.write("죄송합니다 댓글이 없습니다")
    else:
        happy = df_2["label"].value_counts()[1]
        sad = df_2["label"].value_counts()[-1]
        neutral = df_2["label"].value_counts()[0]
        labels = ['positive', 'negative', 'neutral']
        sizes = [happy, sad, neutral]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  
        #ax.text(-2, 0.5, f'positive: {happy}', ha='center', va='center')
        #ax.text(-2, 0.3, f'negative: {sad}', ha='center', va='center')
        #ax.text(-2, 0.1, f'neutral: {neutral}', ha='center', va='center')
        ax.legend(title="Legend", loc="best", bbox_to_anchor=(1, 0, 0.5, 1))
        st.pyplot(fig)
        df_2



st.title("Top30일 주가 예측 모델")
st.markdown("----")

makemodel()