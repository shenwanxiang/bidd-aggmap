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


def GlobalIMP(clf, mp, X, Y, task_type = 'classification', 
              binary_task = False,
              sigmoidy = False, 
              apply_logrithm = False,
              apply_smoothing = False, 
              kernel_size = 5, 
              sigma = 1.6):
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
    df_grid = mp.df_grid_reshape
    backgroud = mp.transform_mpX_to_df(clf.X_).min().values #min value in the training set

    dfY = pd.DataFrame(Y)
    Y_true = Y
    Y_prob = clf._model.predict(X)
    N, W, H, C = X.shape
    T = len(df_grid)
    nX = 20 # 10 arrX to predict

    if (sigmoidy) & (task_type == 'classification'):
        Y_prob = sigmoid(Y_prob)
    

    final_res = {}
    for k, col in enumerate(dfY.columns):
        if (task_type == 'classification') & (binary_task):
            if k == 0:
                continue
        print('calculating feature importance for column %s ...' % col)
        results = []
        loss = f(Y_true[:, k].tolist(), Y_prob[:, k].tolist())
        
        tmp_X = []
        flag = 0
        for i in tqdm(range(T), ascii= True):
            ts = df_grid.iloc[i]
            y = ts.y
            x = ts.x
            
            ## step 1: make permutaions
            X1 = np.array(X)
            #x_min = X[:, y, x,:].min()
            vmin = backgroud[i]
            X1[:, y, x,:] = np.full(X1[:, y, x,:].shape, fill_value = vmin)
            tmp_X.append(X1)

            if (flag == nX) | (i == T-1):
                X2p = np.concatenate(tmp_X)
                ## step 2: make predictions
                Y_pred_prob = clf._model.predict(X2p) #predict ont by one is not efficiency
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

        ## step 4:apply scaling or smothing 
        s = pd.DataFrame(results).values
        if apply_logrithm:
            s = np.log(s)
        smin = np.nanmin(s[s != -np.inf])
        smax = np.nanmax(s[s != np.inf])
        s = np.nan_to_num(s, nan=smin, posinf=smax, neginf=smin) #fillna with smin
        a = scaler.fit_transform(s)
        a = a.reshape(*mp._S.fmap_shape)
        if apply_smoothing:
            covda = conv2(a, kernel_size=kernel_size, sigma=sigma)
            results = covda.reshape(-1,).tolist()
        else:
            results = a.reshape(-1,).tolist()

        final_res.update({col:results})
        
    df = pd.DataFrame(final_res)
    df.columns = df.columns.astype(str)
    df.columns = 'col_' + df.columns + '_importance'
    df = df_grid.join(df)
    return df



def LocalIMP(clf, mp, X, Y, 
             task_type = 'classification', 
             binary_task = False,
             sigmoidy = False,  
             apply_logrithm = False, 
             apply_smoothing = False,
             kernel_size = 3, sigma = 1.2):
    '''
    Forward prop. Feature importance
    '''
    
    assert len(X) == 1, 'each for only one image!'
    
    if task_type == 'classification':
        f = log_loss
    else:
        f = mean_squared_error
        
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    scaler = StandardScaler()
    df_grid = mp.df_grid_reshape
    backgroud = mp.transform_mpX_to_df(clf.X_).min().values #min value in the training set
    
    Y_true = Y
    Y_prob = clf._model.predict(X)
    N, W, H, C = X.shape

    if (sigmoidy) & (task_type == 'classification'):
        Y_prob = sigmoid(Y_prob)

    results = []
    loss = f(Y_true.ravel().tolist(),  Y_prob.ravel().tolist())
    
    all_X1 = []
    for i in tqdm(range(len(df_grid)), ascii= True):
        ts = df_grid.iloc[i]
        y = ts.y
        x = ts.x
        X1 = np.array(X)
        vmin = backgroud[i]
        X1[:, y, x,:] = np.full(X1[:, y, x,:].shape, fill_value = vmin)
        all_X1.append(X1)
        
    all_X = np.concatenate(all_X1)
    all_Y_pred_prob = clf._model.predict(all_X)

    for Y_pred_prob in all_Y_pred_prob:
        if (sigmoidy) & (task_type == 'classification'):
            Y_pred_prob = sigmoid(Y_pred_prob)
        mut_loss = f(Y_true.ravel().tolist(), Y_pred_prob.ravel().tolist()) 
        res =  mut_loss - loss # if res > 0, important, othervise, not important
        results.append(res)

    ## apply smothing and scalings
    s = pd.DataFrame(results).values
    if apply_logrithm:
        s = np.log(s)
    smin = np.nanmin(s[s != -np.inf])
    smax = np.nanmax(s[s != np.inf])
    s = np.nan_to_num(s, nan=smin, posinf=smax, neginf=smin) #fillna with smin
    a = scaler.fit_transform(s)
    a = a.reshape(*mp._S.fmap_shape)
    if apply_smoothing:
        covda = conv2(a, kernel_size=kernel_size, sigma=sigma)
        results = covda.reshape(-1,).tolist()
    else:
        results = a.reshape(-1,).tolist()

    df = pd.DataFrame(results, columns = ['imp'])
    #df.columns = df.columns + '_importance'
    df = df_grid.join(df)
    return df