import os
import pandas as pd
import numpy as np

from glob import glob
from tqdm import tqdm

import cv2
from PIL import Image

from pathlib import Path
from joblib import Parallel, delayed


def price_scaling(dataframe, size):
    if np.nanmin(dataframe) == np.nanmax(dataframe):
        return np.round((dataframe) / (np.nanmax(dataframe)) * size//2,0)
    else:
        return np.round((dataframe - np.nanmin(dataframe)) / (np.nanmax(dataframe) - np.nanmin(dataframe)) * size,0)
        
def vol_scaling(dataframe,size):
    dataframe = dataframe.copy()

    if dataframe.max().values == 0:
        dataframe['VOL'] = [0] * len(dataframe)
        return dataframe
    
    scaled_vol = (np.ceil((dataframe/dataframe.max())/(1/size))).astype(int)
    return scaled_vol

def pickle_to_grayimage(pkl_path):

    df = pd.read_pickle(pkl_path)

    if len(df[~df['CLOSE'].isna()]) == 0 :
        return

    df = df[df[~df['CLOSE'].isna()].index[0]:].copy()

    if len(df) <25:
        return

    if df.isna().iloc[-1]['VOL']:
        df = df.iloc[:-1].copy()

    df['MA20_CLOSE'] = df['CLOSE'].rolling(window=20,min_periods=1).mean()
    df = df.iloc[19:].copy()

    for i in range(20,len(df) + 1):
        
        temp = df.iloc[i-20:i].copy()
        
        if temp['VOL'].isna().sum() > 0 :
            continue

        if temp['MA20_CLOSE'].isna().sum() > 0 :
            continue

        scaled_price = price_scaling(temp[['OPEN','HIGH','LOW','CLOSE','MA20_CLOSE']],50)
        scaled_vol = vol_scaling(temp[['VOL']],12)

        width = 60
        height = 12 + 1
        vol_matrix = np.zeros((height, width), dtype=np.uint8)

        for i in range(20):
            vol_matrix[0:scaled_vol['VOL'].iloc[i], 3*i + 1] = 1

        vol_matrix = np.flipud(vol_matrix)

        idx_lst = np.array([ 3*i+1 for i in range(20)])

        height = 51
        price_matrix = np.zeros((height, width), dtype=np.uint8)

        o = 50 - scaled_price['OPEN']
        h = 50 - scaled_price['HIGH']
        l = 50 - scaled_price['LOW']
        c = 50 - scaled_price['CLOSE']

        price_matrix[np.array(o[~o.isna()].astype(int)),idx_lst[~o.isna()]-1] = 1
        price_matrix[np.array(c[~c.isna()].astype(int)),idx_lst[~c.isna()]+1] = 1

        not_na_h = h[~(h.isna() | l.isna())].astype(int).values
        not_na_l = l[~(h.isna() | l.isna())].astype(int).values
        not_na_idx = idx_lst[~(h.isna() | l.isna())]

        for i in range(len(not_na_idx)):
            price_matrix[not_na_h[i]:not_na_l[i]+1, not_na_idx[i]] = 1

        ma = 50 - scaled_price['MA20_CLOSE'].astype(int).values

        for i in range(20):
            if i < 19:
                cv2.line(price_matrix, (idx_lst[i],ma[i]), (idx_lst[i+1],ma[i+1]), 1, 1)

        matrix = np.vstack((price_matrix, vol_matrix))
        candle_image = Image.fromarray(np.uint8(matrix*255))
        
        path = 'data/image/gray/' + str(temp.index[-1]).split(' ')[0] + '/' + pkl_path.split('/')[-1].split('.pkl')[0] + '.png'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        candle_image.save(path)

if __name__ == '__main__':
    stock_list = glob('data/stock/*')
    stock_list.sort()
 
    with Parallel(n_jobs=-1, verbose=0, backend='loky') as parallel:
        parallel(delayed(pickle_to_grayimage)(Path(path).as_posix()) for path in tqdm(stock_list[:])) 