import os

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from joblib import Parallel, delayed
import os

model = 'ViT_Cap_VW'

port = pd.read_csv(r'Factor_port\\' + model + '.csv', index_col=0)
port.index = pd.to_datetime(port.index)

start_date = pd.to_datetime('2001-01-01')
end_date = pd.to_datetime('2024-12-31')

port = port[(port.index >= start_date) & (port.index <= end_date)]

ff3 = pd.read_csv(r'data\processed_kelly\ff3.csv', index_col=0)
ff3.index = pd.to_datetime(ff3.index)
ff3 = ff3[start_date:end_date]

ff5 = pd.read_csv(r'data\processed_kelly\ff5.csv', index_col=0)
ff5.index = pd.to_datetime(ff5.index)
ff5 = ff5[start_date:end_date]

q5 = pd.read_csv(r'data\processed_kelly\q5.csv')
q5.set_index('date', inplace=True)
q5.index = pd.to_datetime(q5.index)
q5 = q5[start_date:end_date]
q5.index.name = None

test_portfolios = pd.read_csv(r'data\processed_kelly\test_portfolios.csv')
test_portfolios.set_index('date', inplace=True)
test_portfolios.index = pd.to_datetime(test_portfolios.index)
test_portfolios = test_portfolios[start_date:end_date]
test_portfolios = test_portfolios
test_portfolios.index.name = None

mom = pd.read_csv(r'data\processed_kelly\ff_mom.csv', index_col=0)
mom.index = pd.to_datetime(mom.index)
mom = mom[start_date:end_date]

kelly_factors = pd.read_csv(r'data\processed_kelly\kelly_factor.csv')
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

factor_zoo = pd.concat([np.round(kelly_factors*100,4) ,ff5[['Mkt-RF','SMB','HML','RMW','CMA']],q5[['R_ME','R_IA','R_ROE']],np.round(port[[model.split('_')[0]]] * 100 ,4)],axis=1)

h_t = factor_zoo.iloc[:,:-1].copy()
h_t = np.array(h_t)
g_t = np.array(factor_zoo.iloc[:,-1:])

beta_k_df = pd.read_csv(r'data//beta_k.csv', index_col=0)
mean_betas = (beta_k_df**2).mean(axis=0)
penalty = (mean_betas/mean_betas.mean())

def run_one_seed(seed):

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    lasso1_mse_lst = []
    lasso2_mse_lst = []
    
    NumLambda = 100

    lambdas1 = np.exp(np.linspace(0, -35, NumLambda))
    lambdas2 = np.exp(np.linspace(0, -35, NumLambda))

    kf = KFold(n_splits=10, shuffle=True, random_state=seed)

    for fold, (train_index, test_index) in enumerate(kf.split(h_t)):
        lasso1_mse = []
        lasso2_mse = []

        # First Lasso 
     
        h_train = h_t[train_index]
        h_test = h_t[test_index]
    
        y_train = np.array(test_portfolios.iloc[train_index])
        y_test = np.array(test_portfolios.iloc[test_index])

        train_lasso1_cov_lst = []
        
        for idx in range(y_train.shape[1]):
            y_i = y_train[:, idx]
            not_nan_idx = ~np.isnan(y_i)
            y_i = y_i[not_nan_idx]
            x = h_train[not_nan_idx]
            train_lasso1_cov_lst.append(np.cov(y_i,x.T,ddof=1)[0,1:])

        train_lasso1_cov = np.array(train_lasso1_cov_lst)
        mean_Ri_train = np.nanmean(y_train,axis=0).reshape(-1,1)
        
        # -------

        test_lasso1_cov_lst = []
        for idx in range(y_test.shape[1]):
            y_i = y_test[:, idx]
            not_nan_idx = ~np.isnan(y_i)
            y_i = y_i[not_nan_idx]
            x = h_test[not_nan_idx]
            test_lasso1_cov_lst.append(np.cov(y_i,x.T,ddof=1)[0,1:])

        test_lasso1_cov = np.array(test_lasso1_cov_lst)
        mean_Ri_test = np.nanmean(y_test,axis=0).reshape(-1,1)

        # Second Lasso
        
        g_train = g_t[train_index].reshape(-1)
        g_test = g_t[test_index].reshape(-1)

        # just cov array for g

        train_gt_cov_lst =[]

        for idx in range(y_train.shape[1]):
            y_i = y_train[:, idx]
            not_nan_idx = ~np.isnan(y_i)
            y_i = y_i[not_nan_idx]
            x = g_train[not_nan_idx]
            train_gt_cov_lst.append(np.cov(y_i,x.T,ddof=1)[0,1:])

        train_gt_cov = np.array(train_gt_cov_lst).reshape(-1)

        test_gt_cov_lst = []
        for idx in range(y_test.shape[1]):
            y_i = y_test[:, idx]
            not_nan_idx = ~np.isnan(y_i)
            y_i = y_i[not_nan_idx]
            x = g_test[not_nan_idx]
            test_gt_cov_lst.append(np.cov(y_i,x.T,ddof=1)[0,1:])

        test_gt_cov = np.array(test_gt_cov_lst).reshape(-1)

        # lasso 1
        for lambda1 in lambdas1:
            lasso1 = Lasso(alpha=lambda1, fit_intercept=True, max_iter=10000,tol=1e-5)
            lasso1.fit(train_lasso1_cov * penalty.values, mean_Ri_train)

            lasso1_pred = lasso1.predict(test_lasso1_cov * penalty.values)
            lasso1_mse.append(np.mean((lasso1_pred - mean_Ri_test) ** 2))

        # lasso 2
        for lambda2 in lambdas2:
            lasso2 = Lasso(alpha=lambda2, fit_intercept=True, max_iter=10000, tol=1e-5)
            lasso2.fit(train_lasso1_cov * penalty.values, train_gt_cov)
            lasso2_pred = lasso2.predict(test_lasso1_cov * penalty.values)
            lasso2_mse.append(np.mean((lasso2_pred - test_gt_cov) ** 2))

        lasso1_mse_lst.append(lasso1_mse)
        lasso2_mse_lst.append(lasso2_mse)

    lasso1_mse_arr = pd.DataFrame(lasso1_mse_lst)
    lasso2_mse_arr = pd.DataFrame(lasso2_mse_lst)

    os.makedirs(r'result//'+ model , exist_ok=True)

    lasso1_mse_arr.to_csv(r'result//'+ model  + '//' + 'lasso1_mse_seed' + str(seed)  +'.csv', index=False)
    lasso2_mse_arr.to_csv(r'result//'+ model  + '//' + 'lasso2_mse_seed' + str(seed)  +'.csv', index=False)

def main():
    results = Parallel(n_jobs=-1, verbose=-1)(
    delayed(run_one_seed)(seed) for seed in tqdm(range(64,200,1))
) 
    
if __name__ == '__main__':    
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    main()