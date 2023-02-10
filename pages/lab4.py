import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_name):
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    else:
        st.write("The file '{}' doesn't exist.".format(file_name))
        return None

data = load_data(r"mnt\C:\Users\user\Wallpapers\python\samsung.csv")

if data is not None:
    selected_columns = st.multiselect("Select columns to plot", data.columns)
    if len(selected_columns) > 0:
        st.line_chart(data[selected_columns])