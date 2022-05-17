# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:54:38 2021

@author: wanxiang.shen@u.nus.edu
"""


import numpy as np
import pandas as pd

from tqdm import tqdm
from copy import copy

from aggmap.utils.matrixopt import conv2
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.preprocessing import StandardScaler


def islice(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def CalcFeatImp(model, mp, arrX, dfY, task_type = 'classification', 
                sigmoidy = False, apply_scale_smothing = True, kernel_size = 5, sigma = 1.6):
    '''
    Forward prop. Feature importance
    
    apply_scale_smothing: alpplying a smothing on the map
    
    '''
    
    if task_type == 'classification':
        f = log_loss
    else:
        f = mean_squared_error
        
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    scaler = StandardScaler()
    grid = mp.plot_grid()
    Y_true = dfY.values
    df_grid = mp.df_grid.sort_values(['y', 'x']).reset_index(drop=True)
    Y_prob = model.predict(arrX)
    N, W, H, C = arrX.shape
    T = len(df_grid)
    nX = 20 # 10 arrX to predict
    vmin = arrX.min()
    
    if (sigmoidy) & (task_type == 'classification'):
        Y_prob = sigmoid(Y_prob)
    
    final_res = {}
    for k, col in enumerate(dfY.columns):
        if col == 'Healthy':
            continue # omit this feature imp

        print('calculating feature importance for %s ...' % col)
        
        results = []
        loss = f(Y_true[:, k].tolist(), Y_prob[:, k].tolist())
        
        tmp_X = []
        flag = 0
        for i in tqdm(range(T), ascii= True):
            
            ts = df_grid.iloc[i]
            y = ts.y
            x = ts.x
            
            ## step 1: make permutaions
            X1 = np.array(arrX)
            X1[:, y, x,:] = np.full(X1[:, y, x,:].shape, fill_value = arrX.min())
            tmp_X.append(X1)

            if (flag == nX) | (i == T-1):
                X2p = np.concatenate(tmp_X)
                ## step 2: make predictions
                Y_pred_prob = model.predict(X2p) #predict ont by one is not efficiency
                if (sigmoidy) & (task_type == 'classification'):
                    Y_pred_prob = sigmoid(Y_pred_prob)    

                ## step 3: calculate changes
                for Y_pred in islice(Y_pred_prob, N):
                    mut_loss = f(Y_true[:, k].tolist(), Y_pred[:, k].tolist()) 
                    res =  mut_loss - loss # if res > 0, important, othervise, not important
                    results.append(res)

                flag = 0
                tmp_X = []
            flag += 1

        ## step 4: apply smothing and scalings
        if apply_scale_smothing:
            s = np.log(pd.DataFrame(results)).values
            smin = np.nanmin(s[s != -np.inf])
            smax = np.nanmax(s[s != np.inf])
            s = np.nan_to_num(s, nan=smin, posinf=smax, neginf=smin) #fillna with smin
            a = scaler.fit_transform(s)
            a = a.reshape(*mp.fmap_shape)
            IMPM = conv2(a, kernel_size=kernel_size, sigma=sigma)
            results = IMPM.reshape(-1,).tolist()

        final_res.update({col:results})
        
    df = pd.DataFrame(final_res)
    df.columns = df.columns + '_importance'
    df = df_grid.join(df)
    return df