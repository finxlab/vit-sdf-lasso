import os

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
    
from glob import glob
from tqdm import tqdm

model = 'vit_Cap_VW' 
port = pd.read_csv(r'Factor_port//' + model + '.csv', index_col=0)
port.index = pd.to_datetime(port.index)

start_date = pd.to_datetime('2001-01-01')
end_date = pd.to_datetime('2024-12-31')

port = port[(port.index >= start_date) & (port.index <= end_date)]

ff3 = pd.read_csv(r'data/processed_kelly/ff3.csv', index_col=0)
ff3.index = pd.to_datetime(ff3.index)
ff3 = ff3[start_date:end_date]

ff5 = pd.read_csv(r'data/processed_kelly/ff5.csv', index_col=0)
ff5.index = pd.to_datetime(ff5.index)
ff5 = ff5[start_date:end_date]

q5 = pd.read_csv(r'data/processed_kelly/q5.csv')
q5.set_index('date', inplace=True)
q5.index = pd.to_datetime(q5.index)
q5 = q5[start_date:end_date]
q5.index.name = None

test_portfolios = pd.read_csv(r'data/processed_kelly/test_portfolios.csv')
test_portfolios.set_index('date', inplace=True)
test_portfolios.index = pd.to_datetime(test_portfolios.index)
test_portfolios = test_portfolios[start_date:end_date]
test_portfolios = test_portfolios
test_portfolios.index.name = None

mom = pd.read_csv(r'data/processed_kelly/ff_mom.csv', index_col=0)
mom.index = pd.to_datetime(mom.index)
mom = mom[start_date:end_date]

kelly_factors = pd.read_csv(r'data/processed_kelly/kelly_factor.csv')
kelly_factors.set_index('date', inplace=True)
kelly_factors.index = pd.to_datetime(kelly_factors.index)
kelly_factors = kelly_factors[start_date:end_date]
kelly_factors.index.name = None

ff3.index = port.index
ff5.index = port.index
q5.index = port.index
test_portfolios.index = port.index
mom.index = port.index
kelly_factors.index = port.index
rf = ff3['RF']

factor_zoo = pd.concat([np.round(kelly_factors*100,4) ,ff5[['Mkt-RF','SMB','HML','RMW','CMA']],q5[['R_ME','R_IA','R_ROE']]],axis=1)

# === univariate beta calculation ===

beta_k_lst = []
for factor_name in tqdm(factor_zoo.columns):
    x_i = factor_zoo[factor_name]
    betas = []
    
    for asset in test_portfolios.columns:
        y_i = test_portfolios[asset]
        if y_i.isna().sum() > 0:
            y_i = y_i[~y_i.isna()].copy()
            x_i_temp = x_i.loc[y_i.index].copy()
            x_i_temp = sm.add_constant(x_i_temp)
            ols_model = sm.OLS(y_i, x_i_temp).fit()
        else:
            x_i = sm.add_constant(x_i)
            ols_model = sm.OLS(y_i, x_i).fit()

        beta_i = ols_model.params[factor_name]
        betas.append(beta_i)

    beta_k = betas

    beta_k_lst.append(beta_k)

pd.DataFrame(beta_k_lst, index=list(factor_zoo.columns[:-1])).T.to_csv('data/beta_k.csv')