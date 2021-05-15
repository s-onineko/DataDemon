# -*- coding: utf-8 -*-
import streamlit as st
import pandas
import numpy
import warnings
warnings.filterwarnings("ignore")
from pycaret.classification import *
from sklearn.datasets import load_iris
import pandas as pd
import os
import datetime
import base64
import json
import pickle
import uuid
import re


st.title('Data Demon')

##########################################################################################
#                                   file_download_button                                 #
##########################################################################################
def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None
    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=True)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)
    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()
    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)
    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'
    return dl_link

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

##########################################################################################
#                                         Pycaret                                        #
##########################################################################################

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=["target"])
df = pd.concat([X,y], axis=1)

if st.checkbox('テストデータをダウンロードするにはチェックを入れてください'):
    st.write('ファイル名を入力後、ダウンロードボタンを押してください。ダウンロードしたファイルを用いてテストすることができます。')
    # Enter text for testing
    s = 'pd.DataFrame'
    filename = st.text_input('Enter output filename and ext (e.g. my-question.csv, )', 'my-question.csv')
    #pickle_it = st.checkbox('Save as pickle file')
    sample_dtypes = {'list': [1,'a', [2, 'c'], {'b': 2}],
                     'str': 'Hello Streamlit!',
                     'int': 17,
                     'float': 17.0,
                     'dict': {1: 'a', 'x': [2, 'c'], 2: {'b': 2}},
                     'bool': True,
                     'pd.DataFrame': df}
    sample_dtypes = sample_dtypes

    # Download sample
    download_button_str = download_button(sample_dtypes[s], filename, f'Click here to download {filename}', pickle_it=False)
    st.markdown(download_button_str, unsafe_allow_html=True)

    
df_test = st.file_uploader("分析用のCSVファイルの読み込み",type = "csv")
df = pd.read_csv(df_test)

st.dataframe(df.head())
# 前処理
target = st.selectbox("目的変数を選択してください",list(df.columns))
features = st.multiselect("説明変数を選択してください",list(df.columns))
df_list = list(target)
df_list = target + features
st.write(df_list)
df = df.loc[:,df_list]

exp1 = setup(df, target = 'target')
st.table(compare_models())
