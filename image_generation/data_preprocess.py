import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob

crsp = pd.read_csv('data/crsp.csv') # NEED TO Download from CRSP

######## EXCHCD
# 1	New York Stock Exchange
# 2	American Stock Exchange
# 3	The Nasdaq Stock Market

######## SHRCD
# 10,11 common stocks

crsp = crsp[crsp['EXCHCD'].isin([1,2,3])]
crsp = crsp[crsp['SHRCD'].isin([10,11])]

crsp.reset_index(inplace=True, drop=True)
crsp.drop(columns=['EXCHCD', 'SHRCD'], inplace=True, axis=1)

crsp['OPENPRC'] = crsp['OPENPRC'].astype(float)
crsp['ASKHI'] = crsp['ASKHI'].astype(float)
crsp['BIDLO'] = crsp['BIDLO'].astype(float)
crsp['PRC'] = crsp['PRC'].astype(float)
crsp['VOL'] = crsp['VOL'].astype(float)
crsp['RET'] = crsp['RET'].replace('C',np.nan).astype(float)
crsp['PERMNO'] = crsp['PERMNO'].astype(int)

crsp.loc[crsp['PRC'] < 0, 'PRC'] = np.nan
crsp.date = pd.to_datetime(crsp.date)

no_lst = list(crsp['PERMNO'].unique())
no_lst.sort()
date_lst = list(crsp['date'].unique())
date_lst.sort()

df_left = pd.DataFrame(date_lst)
df_left.columns = ['date']
df_left.set_index('date',inplace=True)
df_left.index = pd.to_datetime(df_left.index)

for no in tqdm(no_lst):

    temp = crsp[crsp['PERMNO'] == no].reset_index(drop=True)
    temp = temp.sort_values(by='date')

    temp = temp[['date','OPENPRC','ASKHI','BIDLO','PRC','VOL','RET','SHROUT']].copy()
    temp.loc[0,'RET'] = np.nan

    temp['ADJ_PRICE'] = ((temp['RET']+1) * 1).cumprod()
    temp.loc[0,'ADJ_PRICE'] = 1
    
    temp['OPEN/PRC_ratio'] = temp['OPENPRC'] / temp['PRC']
    temp['HIGH/PRC_ratio'] = temp['ASKHI'] / temp['PRC']
    temp['LOW/PRC_ratio'] = temp['BIDLO'] / temp['PRC']

    temp['ADJ_OPEN'] = temp['OPEN/PRC_ratio'] * temp['ADJ_PRICE']
    temp['ADJ_HIGH'] = temp['HIGH/PRC_ratio'] * temp['ADJ_PRICE']
    temp['ADJ_LOW'] = temp['LOW/PRC_ratio'] * temp['ADJ_PRICE']

    temp['ADJ_OPEN'] = ((temp['VOL'] != 0) * 1).replace(0,np.nan) * temp['ADJ_OPEN']
    temp['ADJ_HIGH'] = ((temp['VOL'] != 0) * 1).replace(0,np.nan) * temp['ADJ_HIGH']
    temp['ADJ_LOW'] = ((temp['VOL'] != 0) * 1).replace(0,np.nan) * temp['ADJ_LOW']
    temp['ADJ_PRICE'] = ((temp['VOL'] != 0) * 1).replace(0,np.nan) * temp['ADJ_PRICE']

    temp = temp[['date','ADJ_OPEN','ADJ_HIGH','ADJ_LOW','ADJ_PRICE','VOL']].copy()
    temp.columns = ['date','OPEN','HIGH','LOW','CLOSE','VOL']
    temp.set_index('date',inplace=True)
    temp.index = pd.to_datetime(temp.index)
    
    os.makedirs('data/stock',exist_ok=True)
    temp.to_pickle(f'data/stock/{no}.pkl')

stk_lst = glob('data/stock/*')
stk_lst.sort()

close_dfs = []

for stk in tqdm(stk_lst):
    df = pd.read_pickle(stk)
    df.index = pd.to_datetime(df.index)
    no = stk.split('/')[-1].split('.pkl')[0]
    no_lst.append(no)
    close_dfs.append(df[['CLOSE']].rename(columns = {'CLOSE': no })[no])

close = pd.concat(close_dfs,axis=1)

close.to_pickle('data/close.pkl')