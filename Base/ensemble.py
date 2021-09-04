###################
# import packages #
###################

import pandas as pd
import glob


#############
# Functions #
#############

# CSV를 불러와 결과를 Hard Voting하는 함수 입니다.
def Hardvoting(data_dir,save_file_name):
    # 결과 CSV path
    all_files = glob.glob(data_dir +"*.csv") 
    li = []
    
    # file path에 따라 파일 로드
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        # 결과 저장
        li.append(df['ans'])
    
    # 결과 종합 concatnate
    frame = pd.concat(li, axis=1, ignore_index=True)

    df['ans'] = frame.mode(axis=1)[0]
    df['ans'] = df['ans'].astype(int)
    df.to_csv(data_dir+save_file_name+'./save_file_name.csv')
                          
# 라벨을 Encoding 하는 함수 입니다.
def Make_label(data_dir,save_file_name,label_list):
    all_files = glob.glob(data_dir +"*.csv") 
    li = []
    df = pd.read_csv(data_dir+'/sample_sumbmission.csv', index_col=None, header=0)        

    for filename,label in zip(all_files,label_list):
        frame = pd.read_csv(filename, index_col=None, header=0)
        df[label] = frame['ans']

    df['ans'] = df['age'] + df['gender'] *3 + df['mask']*6
    df.to_csv(data_dir+save_file_name+'./save_file_name.csv')
