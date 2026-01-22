import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob

crsp = pd.read_csv('data/crsp.csv')

crsp = crsp[crsp['SHRCD'].isin([10,11])].copy()
crsp = crsp[crsp['EXCHCD'].isin([1,2,3])].copy()

crsp.reset_index(drop=True, inplace=True)
crsp['CAP'] = crsp['PRC'].abs() * crsp['SHROUT']
crsp['RET'] = crsp['RET'].replace('C', np.nan).copy()
crsp['RET'] = crsp['RET'].astype(float).copy()
crsp.drop(columns=['SHRCD'], inplace=True)

close_lst =[]
vol_lst = []
cap_lst = []
nyse_cap_lst = []

for no in tqdm(crsp['PERMNO'].unique()):
    temp = crsp[crsp['PERMNO'] == no].copy()
    temp = temp.set_index('date')
    temp.index.name = None
    temp.index=pd.to_datetime(temp.index)
    temp = temp.sort_index()

    nyse_temp = temp[temp['EXCHCD'] == 1].copy()

    if len(nyse_temp) == 0:
        pass
    else:
        nyse_cap = nyse_temp['CAP'].copy()
        nyse_cap.name = no
        nyse_cap.dropna(inplace=True)
        nyse_cap_lst.append(nyse_cap)
    
    cap = temp['CAP'].copy()
    cap.name = no
    cap.dropna(inplace=True)
    cap_lst.append(cap)

    vol = temp['VOL'].copy()
    vol.name = no
    vol.dropna(inplace=True)
    vol_lst.append(vol)

    ret = temp['RET'].copy()
    ret.iloc[0] = 0
    close = (ret+1).cumprod()
    close.name = no
    close.dropna(inplace=True)
    close_lst.append(close)

pd.concat(nyse_cap_lst,axis=1).to_pickle('data/nyse_cap.pkl')
pd.concat(cap_lst, axis=1).to_pickle('data/cap.pkl')
pd.concat(vol_lst, axis=1).to_pickle('data/vol.pkl')
pd.concat(close_lst, axis=1).to_pickle('data/close.pkl')