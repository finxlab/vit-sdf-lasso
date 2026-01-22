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
        return np.round((dataframe) / (np.nanmax(dataframe)) * size//2,0).astype(int)
    else:
        return np.round((dataframe - np.nanmin(dataframe)) / (np.nanmax(dataframe) - np.nanmin(dataframe)) * size,0).astype(int)
        
def vol_scaling(dataframe,size):
    dataframe = dataframe.copy()

    if dataframe.max().values == 0:
        dataframe['VOL'] = [0] * len(dataframe)
        return dataframe
    
    scaled_vol = (np.ceil((dataframe/dataframe.max())/(1/size))).astype(int)
    return scaled_vol

def pickle_to_rgbimage(csv_path):

    df = pd.read_pickle(csv_path)

    if len(df[~df['CLOSE'].isna()]) == 0 :
        return

    df = df[df[~df['CLOSE'].isna()].index[0]:].copy()

    if len(df) < 45:
        return

    if df.isna().iloc[-1]['VOL']:
        df = df.iloc[:-1].copy()

    df['CLOSE'] = df['CLOSE'].ffill()
    df['OPEN'] = df['OPEN'].fillna(df['CLOSE'].shift(1))
    df['HIGH'] = df['HIGH'].fillna(df['CLOSE'])
    df['LOW'] = df['LOW'].fillna(df['CLOSE'])

    df['MA20_CLOSE'] = df['CLOSE'].rolling(window=20,min_periods=1).mean()

    conditions = [
        (df['CLOSE'] - df['OPEN'] > 0),
        (df['CLOSE'] - df['OPEN'] < 0),
        (df['CLOSE'] - df['OPEN'] == 0)
    ]
    choices = ['green', 'red', 'gray']
    df['COLOR'] = np.select(conditions, choices, default='nan')

    df = df.iloc[19:].copy()

    for i in range(25,len(df)+1):

        temp = df.iloc[i-25:i].copy()

        if len(temp) < 25:
            continue

        if temp['VOL'].isna().sum() > 0 : 
            continue

        scaled_price = price_scaling(temp[['OPEN','HIGH','LOW','CLOSE','MA20_CLOSE']],159)
        scaled_vol = vol_scaling(temp[['VOL']],63)

        colors = temp['COLOR'].values

        image_matrix = np.zeros((224, 224, 3), dtype=np.uint8) # y,x,rgb

        candle_width = 6
        space_width = 3

        ma20_matrix = np.zeros((224, 224, 3), dtype=np.uint8) 

        for index in range(len(scaled_price)):
            # MA
            if index < len(scaled_price) - 1:
                MA20_px_today = scaled_price.iloc[index]['MA20_CLOSE'] + 64
                MA20_px_tomorrow = scaled_price.iloc[index + 1]['MA20_CLOSE'] + 64
                
                start_x_today = 1 + index * (candle_width + space_width) + candle_width // 2
                start_x_tomorrow = 1 + (index + 1) * (candle_width + space_width) + candle_width // 2

                cv2.line(ma20_matrix, 
                        (start_x_today, MA20_px_today), 
                        (start_x_tomorrow, MA20_px_tomorrow), 
                        (0, 0, 255), 1)

        blue_pixels = np.where((ma20_matrix == [0, 0, 255]).all(axis=2))

        for y, x in zip(blue_pixels[0], blue_pixels[1]):
            if y > 0:
                ma20_matrix[y-1, x] = [0, 0, 255]
            if y < ma20_matrix.shape[0] - 1:
                ma20_matrix[y+1, x] = [0, 0, 255]

        image_matrix = ma20_matrix

        for index in range(len(scaled_price)):
            open_px = scaled_price.iloc[index]['OPEN'] + 64
            close_px = scaled_price.iloc[index]['CLOSE'] + 64
            high_px = scaled_price.iloc[index]['HIGH'] + 64
            low_px = scaled_price.iloc[index]['LOW'] + 64

            start_box_y = min(open_px, close_px)
            end_box_y = max(open_px, close_px)

            color = colors[index]
            
            start_x = 1 + index * (candle_width + space_width)
            
            if color == 'gray':
                rgb = np.array([64, 64, 64]).astype(np.uint8)

                image_matrix[ start_box_y : end_box_y +1, 
                            start_x : start_x + candle_width, 
                            :] = np.minimum(image_matrix[start_box_y:end_box_y + 1, 
                            start_x:start_x + candle_width, 
                            :] + rgb, 255)

                image_matrix[low_px : high_px +1, 
                            start_x + 2: start_x + candle_width - 2, 
                            :] = np.minimum(image_matrix[low_px:high_px+1, 
                                start_x + 2:start_x + candle_width - 2, 
                                :] + rgb, 255)
            else:
                if color == 'red':
                    rgb = np.array([255, 0, 0]).astype(np.uint8)

                elif color =='green':
                    rgb = np.array([0, 255, 0]).astype(np.uint8)
                
                image_matrix[ start_box_y : end_box_y +1, 
                            start_x : start_x + candle_width, 
                            :] += rgb
                
                image_matrix[low_px : high_px +1, 
                            start_x + 2: start_x + candle_width - 2, 
                            :] += rgb

        for index in range(len(scaled_vol)):

            vol_pox = scaled_vol.iloc[index]['VOL']

            color = colors[index]
            
            if color == 'red':
                rgb = np.array([255, 0, 0]).astype(np.uint8)

            elif color =='green':
                rgb = np.array([0, 255, 0]).astype(np.uint8)

            elif color == 'gray':
                rgb = np.array([64, 64, 64]).astype(np.uint8)

            start_x = 1 + index * (candle_width + space_width)

            image_matrix[ 0 : vol_pox, 
                        start_x : start_x + candle_width, 
                        :] += rgb

        image_matrix = np.flipud(image_matrix)

        image_matrix[image_matrix == 64] = 128 # 64 where gray
        image_matrix[image_matrix == 63] = 192 # overflow with 64 (which should be 128) + 255 ma - > avg value
        image_matrix[image_matrix == 127] = 192 # overflow with 128 (64 + 64 , cross section of gray ver hor) + 255 ma - > avg value
        image_matrix[image_matrix == 254] = 255 # overflow with 255 + 255 cross section of ver hor in green and red 

        image_matrix[160,:,:] = 0 # seperator

        candle_image = Image.fromarray(image_matrix, 'RGB')

        path = 'data/image/rgb/' + str(temp.index[-1]).split(' ')[0] + '/' + csv_path.split('/')[-1].split('.pkl')[0] + '.png'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        candle_image.save(path)

if __name__ == '__main__':
    stock_list = glob('data/stock/*')
    stock_list.sort()
 
    with Parallel(n_jobs=-1, verbose=0, backend='loky') as parallel:
        parallel(delayed(pickle_to_rgbimage)(Path(path).as_posix()) for path in tqdm(stock_list[:])) 