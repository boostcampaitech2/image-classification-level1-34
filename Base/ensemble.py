import pandas as pd
import glob

def Hardvoting(data_dir,save_file_name):
    all_files = glob.glob(data_dir +'*.csv") 
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df['ans'])

    frame = pd.concat(li, axis=1, ignore_index=True)

    df['ans'] = frame.mode(axis=1)[0]
    df['ans'] = df['ans'].astype(int)
    df.to_csv(data_dir+save_file_name'./save_file_name.csv')
                          
def Make_label(data_dir,save_file_name,label_list):
    all_files = glob.glob(data_dir +'*.csv") 
    li = []
    df = pd.read_csv(datadir+'/sample_sumbmission.csv', index_col=None, header=0)        

    for filename,label in zip(all_files,label_list):
        frame = pd.read_csv(filename, index_col=None, header=0)
        df[label] = frame['ans']

    df['ans'] = df['age'] + df['gender'] *3 + df['mask']*6
    df.to_csv(data_dir+save_file_name'./save_file_name.csv')