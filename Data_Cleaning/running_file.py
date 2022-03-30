import pandas as pd
import numpy as np
import os

all_folders = os.listdir('Raw_data')
all_folders = all_folders[1:]


for i in all_folders:
    df_t = pd.DataFrame()
    test_files = os.listdir('Raw_data/{}'.format(i))
    for file in test_files:
        path = 'Raw_data/{}/{}'.format(i,file)
        temp = pd.read_csv(path)
        temp_1 = temp[temp['model'] == 'ST4000DM000']
        names = list(temp_1.columns)
        filter_names = [name for name in names if 'raw' not in name]      
        temp_2 = temp_1[filter_names]
        df_t = pd.concat([df_t, temp_2])
    print(f'Done processing {i}')
    naming = f'Raw_data/Collated/{i}_collated.csv'
    df_t.to_csv(naming)

