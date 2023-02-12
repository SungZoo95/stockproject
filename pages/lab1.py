import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# import cv2

st.title("핫한 30개 기업종목")

file = st.file_uploader("name_code_0206.csv")
#file_path = file
#st.write("The path of the uploaded file is:", file_path)

if file:
    #data = pd.read_csv(file, dtype=str).drop(['Unnamed: 0'], axis=1)
    data = pd.read_csv(file, dtype=str)
    st.write(data)

#df = pd.read_csv('name_code_0206.csv')
#st.dataframe(df)

#if not data.empty:
#    yticks = np.arange(55000, 65000, 500)
#    plt.yticks(yticks)
#    st.line_chart(data=data, x=data["날짜"], y=data["종가"], width=10, height = 15)
    
bar_color = st.color_picker('Pick Bar Color', '#446AF1')
plot_color = st.color_picker('Pick plot Color', '#F90000')
x = data["날짜"]
y = data["종가"]
plt.plot(x,y, color=plot_color, marker='*')
plt.bar(x,y, color=bar_color)
plt.xticks(rotation=45)
st.pyplot()







#if file is None:
#    file = st.camera_input("take picture")
#    # bin 파일을 jpg로 변경하는 코드 
#
#if file is not None:
#  st.download_button(" 사진을 다운로드 ", file)