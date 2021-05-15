# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import datetime
import base64
import pickle
import uuid
import re
import json

from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

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
            object_to_download = object_to_download.to_csv(index=True)
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
#                                      Sampledata                                        #
##########################################################################################
if st.checkbox('テスト用サンプルデータをダウンロードするにはチェックを入れてください'):
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    # Enter text for testing
    s = 'pd.DataFrame'
    sample_dtypes = {'list': [1,'a', [2, 'c'], {'b': 2}],
                     'str': 'Hello Streamlit!',
                     'int': 17,
                     'float': 17.0,
                     'dict': {1: 'a', 'x': [2, 'c'], 2: {'b': 2}},
                     'bool': True,
                     'pd.DataFrame':train_data}
    sample_dtypes = sample_dtypes
    st.write("テストデータは正解ラベル付きです。実際に予測する場合はデータを削除して実施してください")
    # Download sample
    download_button_str = download_button(sample_dtypes[s], "amazon_aws_traindata.csv", 'Click here to download amazon_aws_traindata.csv')
    st.markdown(download_button_str, unsafe_allow_html=True)
    # Enter text for testing
    s = 'pd.DataFrame'
    sample_dtypes = {'list': [1,'a', [2, 'c'], {'b': 2}],
                     'str': 'Hello Streamlit!',
                     'int': 17,
                     'float': 17.0,
                     'dict': {1: 'a', 'x': [2, 'c'], 2: {'b': 2}},
                     'bool': True,
                     'pd.DataFrame':test_data}
    sample_dtypes = sample_dtypes
    # Download sample
    download_button_str = download_button(sample_dtypes[s], "amazon_aws_testdata.csv", 'Click here to download amazon_aws_testdata.csv')
    st.markdown(download_button_str, unsafe_allow_html=True)

##########################################################################################
#                                       Load_dataset                                     #
##########################################################################################
train_data = st.file_uploader("教師データを読み込んでください",type = "csv")
df_train = pd.read_csv(train_data)
train_data, test_data = train_test_split(namelist, test_size = 0.3, random_state = 111)
# test_data = st.file_uploader("テストデータを読み込んでください",type = "csv")
# df_test = pd.read_csv(test_data)

# 読み込んだデータのサマリー
st.dataframe(df_train.head())
label = st.selectbox("目的変数を選択してください",list(df_train.columns))
st.write("Summary of target variable")
st.table(df_train[label].describe())

##########################################################################################
#                                          Test                                          #
##########################################################################################
run_pred = st.checkbox("AutoMLによる予測を実行")

if run_pred == True :
    save_path = 'agModels-predictClass'  # specifies folder to store trained models
    predictor = TabularPredictor(label=label, path=save_path, presets='best_quality', time_limit=600).fit(df_train)
    y_test = df_test[label]  # values to predict
    test_data_nolab = df_test.drop(columns=[label])  # delete label column to prove we're not cheating
    predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file
    y_pred = predictor.predict(test_data_nolab)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    leaderboard = predictor.leaderboard(df_test, silent=True)
    st.dataframe(leaderboard)

    # Enter text for testing
    s = 'pd.DataFrame'
    sample_dtypes = {'list': [1,'a', [2, 'c'], {'b': 2}],
                     'str': 'Hello Streamlit!',
                     'int': 17,
                     'float': 17.0,
                     'dict': {1: 'a', 'x': [2, 'c'], 2: {'b': 2}},
                     'bool': True,
                     'pd.DataFrame':y_pred}
    sample_dtypes = sample_dtypes
    # Download sample
    download_button_str = download_button(sample_dtypes[s], "predictor.csv", 'Click here to download predictor.csv')
    st.markdown(download_button_str, unsafe_allow_html=True)
    st.write(predictor.fit_summary())
    
    #y_predproba = predictor.predict_proba(test_data)
    
else:
    st.write("※チェックを入れると教師データによるモデルの学習とテストデータへの予測結果の反映が行われます")
