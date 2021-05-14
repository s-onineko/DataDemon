# -*- coding: utf-8 -*-
import streamlit as st
import pandas
import numpy

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



df_clst = pd.DataFrame({'q_1' : int(q1),
                        'q_2' : int(q2),
                        'q_3' : int(q3),
                        'q_4' : int(q4),
                        'q_5' : int(q5),
                        'q_6' : int(q6),
                        'q_7' : int(q7),
                        'q_8' : int(q8),
                        'q_9' : int(q9),
                        'q_10' : int(q10),
                        'q_11' : int(q11),
                        'q_12' : int(q12),
                        'q_13' : int(q13),
                        'q_14' : int(q14),
                        'q_15' : int(q15),
                        'q_16' : int(q16),
                        'q_17' : int(q17),
                        'q_18' : int(q18),
                        'q_19' : int(q19),
                        'q_20' : int(q20),
                        'q_21' : int(q21),
                        'q_22' : int(q22),
                        'q_23' : int(q23)},index=['answer',])
dt_now = datetime.datetime.now()
time = dt_now.strftime('%Y%m%d %H%M')        

if st.checkbox('回答結果をダウンロードするにはチェックを入れてください'):
    st.write('ファイル名を入力後、ダウンロードボタンを押してください。ダウンロードしたファイルは「欲求フラグ判定結果」モードでレーダーチャートとして可視化できます。')
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
                     'pd.DataFrame': df_clst}
    sample_dtypes = sample_dtypes

    # Download sample
    download_button_str = download_button(sample_dtypes[s].T, filename, f'Click here to download {filename}', pickle_it=False)
    st.markdown(download_button_str, unsafe_allow_html=True)

