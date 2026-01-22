import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

start_date = pd.to_datetime('2001-01-01')
end_date = pd.to_datetime('2024-12-31')

ff3 = pd.read_csv(r'data/processed_kelly/ff3.csv', index_col=0)
ff3.index = pd.to_datetime(ff3.index)
ff3 = ff3[start_date:end_date]

cap = pd.read_pickle(r'data/cap.pkl')
cls = pd.read_pickle(r'data/close.pkl')
vol = pd.read_pickle(r'data/vol.pkl')
nyse = pd.read_pickle(r'data/nyse_cap.pkl')

cls = cls.ffill()[~vol.isna()].copy()
cap = cap.ffill()[~vol.isna()].copy()

cls = cls[cls.index>='2000-01-01'].dropna(how='all', axis=1)
vol = vol[vol.index>='2000-01-01'].dropna(how='all', axis=1)
cap = cap[cap.index>='2000-01-01'].dropna(how='all', axis=1)
nyse = nyse[nyse.index>='2000-01-01'].dropna(how='all', axis=1)

cls.index = pd.to_datetime(cls.index)
vol.index = pd.to_datetime(vol.index)
cap.index = pd.to_datetime(cap.index)
nyse.index = pd.to_datetime(nyse.index)

nyse_june_data = nyse[nyse.index.month == 6].copy()
nyse_last_june_dates = nyse_june_data.groupby(nyse_june_data.index.year).apply(lambda x: x.loc[x.index.max()])

breakpoint = {}

for year in nyse_last_june_dates.index:
    breakpoint[year] = {}

    temp = nyse_last_june_dates.loc[year].copy()

    breakpoint[year]['NYSE_20th'] = np.percentile(temp.dropna(),20)
    breakpoint[year]['NYSE_40th'] = np.percentile(temp.dropna(),40)
    breakpoint[year]['NYSE_60th'] = np.percentile(temp.dropna(),60)
    breakpoint[year]['NYSE_80th'] = np.percentile(temp.dropna(),80)
    breakpoint[year]['NYSE_50th'] = np.percentile(temp.dropna(),50)

breakpoint_df = pd.DataFrame(breakpoint).T

market_june_data = cap[cap.index.month == 6].copy()
market_last_june_dates = market_june_data.groupby(market_june_data.index.year).apply(lambda x: x.loc[x.index.max()])


ViT = r'pred/vit'
CNN = r'pred/cnn'

model_paths = [ViT, CNN]

for model in model_paths: #model_paths:

    sortportfolio = []

    pred_path = model

    pred_csv_path = glob(pred_path + '/*')
    pred_csv_path.sort()

    rebal_date = [date.split('/')[-1].split('.csv')[0] for date in pred_csv_path]
    
    for idx, path in enumerate(tqdm(pred_csv_path)):
        
        # date -> end of month 

        date = path.split('/')[-1].split('.')[0]
        pred_df = pd.read_csv(path,index_col=0)
        pred_df.index = pred_df.index.astype(int)
        pred_val = pred_df.mean(axis=1)

        nyse_data_at_date = nyse.loc[date].dropna().copy()
        
        all_nyse_lst = nyse_data_at_date.index
        non_micro_lst = nyse_data_at_date[nyse_data_at_date >= np.percentile(nyse_data_at_date,20)].index

        all_nyse_signal_breakpoint = pred_val.loc[pred_val.index[pred_val.index.isin(all_nyse_lst)]].copy()
        non_micro_nyse_signal_breakpoint = pred_val.loc[pred_val.index[pred_val.index.isin(non_micro_lst)]].copy()
        
        year = date.split('-')[0]
        month = date.split('-')[1]

        # Check Date -> sort portfolio before/after June

        if 6 <= int(month):
            pass
        else:
            year = str(int(year) - 1)

        breakpoint = breakpoint_df.loc[int(year)].copy()
        trade_assets_cap = market_last_june_dates.loc[int(year)].copy().dropna()

        cap_vw = np.percentile(nyse.loc[date].dropna(),80)

        temp  = pd.DataFrame(trade_assets_cap)
        temp = pd.concat([temp,pred_val,cap.loc[date]],axis=1).dropna(axis=0)
        temp.columns = ['breakpoint_cap', 'pred','cap']
        temp['winsorized_cap'] = temp['cap'].map(lambda x : np.min([x,cap_vw]))
        temp['date'] = ff3.index[idx]

        temp['size_decile_5'] = (temp['breakpoint_cap'] < breakpoint['NYSE_20th']) * 1 + \
                            ((breakpoint['NYSE_20th'] <= temp['breakpoint_cap']) & (temp['breakpoint_cap'] < breakpoint['NYSE_40th'])) * 2  + \
                            ((breakpoint['NYSE_40th'] <= temp['breakpoint_cap']) & (temp['breakpoint_cap'] < breakpoint['NYSE_60th'])) * 3 + \
                            ((breakpoint['NYSE_60th'] <= temp['breakpoint_cap']) & (temp['breakpoint_cap'] < breakpoint['NYSE_80th'])) * 4 + \
                            ((breakpoint['NYSE_80th'] <= temp['breakpoint_cap'])) * 5
        
        temp['size_decile_3'] = (temp['breakpoint_cap'] < breakpoint['NYSE_20th']) * 1 + \
                            ((breakpoint['NYSE_20th'] <= temp['breakpoint_cap']) & (temp['breakpoint_cap'] < breakpoint['NYSE_50th'])) * 2 + \
                            ((breakpoint['NYSE_50th'] <= temp['breakpoint_cap'])) * 3
        
        temp['signal_decile_5'] = (temp['pred'] < np.percentile(all_nyse_signal_breakpoint,20)) * 1 + \
                            ((np.percentile(all_nyse_signal_breakpoint,20) <= temp['pred']) & (temp['pred'] < np.percentile(all_nyse_signal_breakpoint,40))) * 2 + \
                            ((np.percentile(all_nyse_signal_breakpoint,40) <= temp['pred']) & (temp['pred'] < np.percentile(all_nyse_signal_breakpoint,60))) * 3 + \
                            ((np.percentile(all_nyse_signal_breakpoint,60) <= temp['pred']) & (temp['pred'] < np.percentile(all_nyse_signal_breakpoint,80))) * 4 + \
                            ((np.percentile(all_nyse_signal_breakpoint,80) <= temp['pred'])) * 5
        
        temp['signal_decile_3'] = (temp['pred'] < np.percentile(non_micro_nyse_signal_breakpoint,30)) * 1 + \
                            ((np.percentile(non_micro_nyse_signal_breakpoint,30) <= temp['pred']) & (temp['pred'] < np.percentile(non_micro_nyse_signal_breakpoint,70))) * 2 + \
                            ((np.percentile(non_micro_nyse_signal_breakpoint,70) <= temp['pred'])) * 3
        
        if idx < len(rebal_date)-1:
            price_df = cls[(rebal_date[idx]< cls.index) & (cls.index <= rebal_date[idx+1])].copy()
        else:
            price_df = cls[(rebal_date[idx]< cls.index)].copy()

        trade_price_df = price_df[price_df.columns[price_df.columns.isin(vol.loc[price_df.index[0]][vol.loc[price_df.index[0]]>0].index)]].copy().ffill()
        month_return = pd.DataFrame(((trade_price_df.iloc[-1]-trade_price_df.iloc[0])/ trade_price_df.iloc[0]))
        month_return.columns = ['month_return']

        temp = pd.concat([temp, month_return], axis=1).dropna(axis=0)

        temp['size_decile_5'] = temp['size_decile_5'].astype(int)
        temp['size_decile_3'] = temp['size_decile_3'].astype(int)
        temp['signal_decile_5'] = temp['signal_decile_5'].astype(int)
        temp['signal_decile_3'] = temp['signal_decile_3'].astype(int)

        temp.reset_index(inplace=True)
        temp.rename(columns={'index': 'ticker'}, inplace=True)

        sortportfolio.append(temp)

    sortportfolio_info = pd.concat(sortportfolio)
    sortportfolio_info.reset_index(drop=True, inplace=True)
    sortportfolio_info['date'] = pd.to_datetime(sortportfolio_info['date'])


    fee = 0.0001 * 10 # 10bps

    total_return = []

    for date in tqdm(sortportfolio_info['date'].unique()):

        temp = sortportfolio_info[sortportfolio_info['date'] == date].copy()

        for size in [1,2,3,4,5]:
            for signal in [1,2,3,4,5]:
                temp2 = temp[(temp['size_decile_5'] == size) & (temp['signal_decile_5'] == signal)][['ticker','cap', 'winsorized_cap', 'month_return']].copy()

                if len(temp2) == 0:
                    continue

                temp2['fee_return'] = (1-fee) * (1 + temp2['month_return']) * (1-fee) - 1 # buy fee * return * sell fee

                temp2['cap_VW'] = temp2['winsorized_cap'] / temp2['winsorized_cap'].sum()
                temp2['VW'] = temp2['cap'] / temp2['cap'].sum()
                temp2['EW'] = 1 / len(temp2)

                cap_vw_rt = (temp2['fee_return'] * temp2['cap_VW']).sum()
                vw_rt = (temp2['fee_return'] * temp2['VW']).sum()
                ew_rt = (temp2['fee_return'] * temp2['EW']).sum()

                total_return.append([date, size, signal, cap_vw_rt, vw_rt, ew_rt])

    total_return_df = pd.DataFrame(total_return)
    total_return_df.columns = ['date', 'size_decile_5', 'signal_decile_5','Cap_VW_return' ,'VW_return', 'EW_return']
    total_return_df.to_csv('sorted_portfolio//' + model.split('/')[-1] + '_5by5.csv', index=False)

    total_return = []

    for date in tqdm(sortportfolio_info['date'].unique()):

        temp = sortportfolio_info[sortportfolio_info['date'] == date].copy()

        for size in [1,2,3]:
            for signal in [1,2,3]:
                temp2 = temp[(temp['size_decile_3'] == size) & (temp['signal_decile_3'] == signal)][['ticker','cap', 'winsorized_cap', 'month_return']].copy()

                if len(temp2) == 0:
                    continue

                temp2['fee_return'] = (1-fee) * (1 + temp2['month_return']) * (1-fee) - 1 # buy fee * return * sell fee

                temp2['cap_VW'] = temp2['winsorized_cap'] / temp2['winsorized_cap'].sum()
                temp2['VW'] = temp2['cap'] / temp2['cap'].sum()
                temp2['EW'] = 1 / len(temp2)

                cap_vw_rt = (temp2['fee_return'] * temp2['cap_VW']).sum()
                vw_rt = (temp2['fee_return'] * temp2['VW']).sum()
                ew_rt = (temp2['fee_return'] * temp2['EW']).sum()

                total_return.append([date, size, signal, cap_vw_rt, vw_rt, ew_rt])

    total_return_df = pd.DataFrame(total_return)
    total_return_df.columns = ['date', 'size_decile_3', 'signal_decile_3','Cap_VW_return' ,'VW_return', 'EW_return']
    total_return_df.to_csv('sorted_portfolio//' + model.split('/')[-1] + '_3by3.csv', index=False)




sortportfolio = []

pred_path = r'pred/vit'

pred_csv_path = glob(pred_path + '/*')
pred_csv_path.sort()

rebal_date = [date.split('/')[-1].split('.csv')[0] for date in pred_csv_path]

for idx, path in enumerate(tqdm(pred_csv_path)):
    
    date = path.split('/')[-1].split('.')[0]

    nyse_data_at_date = nyse.loc[date].dropna().copy()
    
    all_nyse_lst = nyse_data_at_date.index
    non_micro_lst = nyse_data_at_date[nyse_data_at_date >= np.percentile(nyse_data_at_date,20)].index

    year = date.split('-')[0]
    month = date.split('-')[1]

    # Check Date -> sort portfolio before/after June

    if 6 <= int(month):
        pass
    else:
        year = str(int(year) - 1)

    breakpoint = breakpoint_df.loc[int(year)].copy()
    trade_assets_cap = market_last_june_dates.loc[int(year)].copy().dropna()

    cap_vw = np.percentile(nyse.loc[date].dropna(),80)

    temp  = pd.DataFrame(trade_assets_cap)
    temp = pd.concat([temp,cap.loc[date]],axis=1).dropna(axis=0)
    temp.columns = ['breakpoint_cap','cap']

    temp['winsorized_cap'] = temp['cap'].map(lambda x : np.min([x,cap_vw]))
    temp['date'] = ff3.index[idx]

    temp['size_decile_3'] = (temp['breakpoint_cap'] < breakpoint['NYSE_20th']) * 1 + \
                        ((breakpoint['NYSE_20th'] <= temp['breakpoint_cap']) & (temp['breakpoint_cap'] < breakpoint['NYSE_50th'])) * 2 + \
                        ((breakpoint['NYSE_50th'] <= temp['breakpoint_cap'])) * 3
    
    if idx < len(rebal_date)-1:
        price_df = cls[(rebal_date[idx]< cls.index) & (cls.index <= rebal_date[idx+1])].copy()
    else:
        price_df = cls[(rebal_date[idx]< cls.index)].copy()

    trade_price_df = price_df[price_df.columns[price_df.columns.isin(vol.loc[price_df.index[0]][vol.loc[price_df.index[0]]>0].index)]].copy().ffill()
    month_return = pd.DataFrame(((trade_price_df.iloc[-1]-trade_price_df.iloc[0])/ trade_price_df.iloc[0]))
    month_return.columns = ['month_return']

    temp = pd.concat([temp, month_return], axis=1).dropna(axis=0)

    temp['size_decile_3'] = temp['size_decile_3'].astype(int)

    temp.reset_index(inplace=True)
    temp.rename(columns={'index': 'ticker'}, inplace=True)
    sortportfolio.append(temp)

sortportfolio_info = pd.concat(sortportfolio)
sortportfolio_info.reset_index(drop=True, inplace=True)
sortportfolio_info['date'] = pd.to_datetime(sortportfolio_info['date'])

fee = 0.0001 * 10 # 10bps

# Cap VW
total_return = []

for date in tqdm(sortportfolio_info['date'].unique()):

    temp = sortportfolio_info[sortportfolio_info['date'] == date].copy()

    temp2 = temp[temp['size_decile_3']!=1].copy()

    if len(temp2) == 0:
        continue

    temp2['fee_return'] = (1-fee) * (1 + temp2['month_return']) * (1-fee) - 1 # buy fee * return * sell fee
    temp2['cap_VW'] = temp2['winsorized_cap'] / temp2['winsorized_cap'].sum()

    cap_vw_rt = (temp2['fee_return'] * temp2['cap_VW']).sum()
    total_return.append([date, cap_vw_rt])


total_return_df = pd.DataFrame(total_return)
total_return_df.columns = ['date','Cap_VW_return']
total_return_df.to_csv('sorted_portfolio//' + 'market.csv', index=False)

# VW
total_return2 = []

for date in tqdm(sortportfolio_info['date'].unique()):

    temp = sortportfolio_info[sortportfolio_info['date'] == date].copy()

    temp2 = temp[temp['size_decile_3']!=1].copy()

    if len(temp2) == 0:
        continue

    temp2['fee_return'] = (1-fee) * (1 + temp2['month_return']) * (1-fee) - 1 # buy fee * return * sell fee
    temp2['VW'] = temp2['cap'] / temp2['cap'].sum()

    vw_rt = (temp2['fee_return'] * temp2['VW']).sum()
    total_return2.append([date, vw_rt])

total_return_df2 = pd.DataFrame(total_return2)
total_return_df2.columns = ['date','VW_return']
total_return_df2.to_csv('sorted_portfolio//' + 'market_VW.csv', index=False)

# EW
total_return3 = []

for date in tqdm(sortportfolio_info['date'].unique()):

    temp = sortportfolio_info[sortportfolio_info['date'] == date].copy()

    temp2 = temp[temp['size_decile_3']!=1].copy()

    if len(temp2) == 0:
        continue

    temp2['fee_return'] = (1-fee) * (1 + temp2['month_return']) * (1-fee) - 1 # buy fee * return * sell fee
    temp2['EW'] = 1 / len(temp2)

    ew_rt = (temp2['fee_return'] * temp2['EW']).sum()
    total_return3.append([date, ew_rt])
    
total_return_df3 = pd.DataFrame(total_return3)
total_return_df3.columns = ['date','EW_return']
total_return_df3.to_csv('sorted_portfolio//' + 'market_EW.csv', index=False)