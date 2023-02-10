import streamlit as st
import pandas as pd
import seaborn as sns

file = st.file_uploader("Upload your file", type=["csv"])

if file:
    data = pd.read_csv(file,dtype=str).drop(['Unnamed: 0'], axis=1)
    st.write(data)
    
selected_columns = st.multiselect("Select columns to plot", data.columns)
if len(selected_columns) > 0:
    st.line_chart(data[selected_columns])