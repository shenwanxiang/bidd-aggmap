import warnings, os
warnings.filterwarnings("ignore")

from copy import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_score, recall_score, f1_score


import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load

from aggmap import AggMap, AggModel, loadmap
from aggmap.AggModel import load_model, save_model


np.random.seed(666) #just for reaptable results

def score(dfr):
    y_true = dfr.y_true
    y_score = dfr.y_score
    y_pred = dfr.y_pred
    '''
    the metrics are taken from orignal paper:
    https://github.com/YDaiLab/Meta-Signer/blob/bd6a1cd98d1035f848ecb6e53d9ee67a85871db2/src/utils/metasigner_io.py#L34
    '''
    auc = roc_auc_score(y_true, y_score, average='weighted')        
    mcc = matthews_corrcoef(y_true, y_pred)
    pres = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print('roc-auc: %.3f, mcc: %.3f, pres: %.3f, recall: %.3f, f1: %.3f' % (auc, mcc, pres, recall, f1))
    return auc, mcc, pres, recall, f1


def get_best_fill_scale(dfx, dfy,  gpuid = 6):

    fill_value_list = [1e-2, 1e-5, 1e-8]
    scale_methods = ['minmax', 'standard']
    all_res = []
    if len(dfx) < 500:
        batch_size = 2
    else:
        batch_size = 8   
    #fill values
    for fill in fill_value_list:
        dfx_new = pd.DataFrame(np.log(dfx + fill))
        mp = AggMap(dfx_new)
        mp = mp.fit(cluster_channels = 5, verbose = 0, var_thr = 0)            
        X = mp.batch_transform(dfx_new.values)
        Y = dfy.values
        outer = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 123)
        outer_idx = outer.split(range(len(dfy)), dfy.idxmax(axis=1))
        for i, idx in enumerate(list(outer_idx)):
            if i in [1, 3, 5, 7, 9]:        
                fold_num = "fold_%s" % str(i).zfill(2) 
                train_idx, test_idx = idx
                testY = Y[test_idx]
                testX = X[test_idx]
                trainX = X[train_idx]
                trainY = Y[train_idx]
                #print("\n %s: input train and test X shape is %s, %s " % (fold_num, trainX.shape,  testX.shape))
                clf = AggModel.MultiClassEstimator(epochs=30, batch_size=batch_size, gpuid = gpuid, 
                                                   monitor='val_loss', patience=1000, verbose=0 ) #
                clf.fit(trainX, trainY, testX, testY)  #,  
                best_epoch = clf._performance.best_epoch + 1
                best_loss = round(clf._performance.best, 3)
                res = {'best_loss':best_loss, 'best_epoch':best_epoch,'fill': fill, 'fold_num':fold_num}
                print(res)
                all_res.append(res)
                
    df = pd.DataFrame(all_res)
    best_fill = df.groupby(['fill'])['best_loss'].mean().idxmin()
    best_epoch1 = df[df.fill == best_fill].best_epoch.median()
    best_epoch1 = int(best_epoch1)
    
    ## scale method
    dfx_new = pd.DataFrame(np.log(dfx + best_fill))
    mp = AggMap(dfx_new)
    mp = mp.fit(cluster_channels = 5, verbose = 0, var_thr = 0)     
    all_res = []
    for scale_method in scale_methods:
        X = mp.batch_transform(dfx_new.values, scale_method = scale_method)
        Y = dfy.values
        outer = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 123)
        outer_idx = outer.split(range(len(dfy)), dfy.idxmax(axis=1))
        for i, idx in enumerate(list(outer_idx)):
            if i in [1, 3, 5, 7, 9]:    
                fold_num = "fold_%s" % str(i).zfill(2) 
                train_idx, test_idx = idx
                testY = Y[test_idx]
                testX = X[test_idx]
                trainX = X[train_idx]
                trainY = Y[train_idx]
                #print("\n %s: input train and test X shape is %s, %s " % (fold_num, trainX.shape,  testX.shape))
                clf = AggModel.MultiClassEstimator(epochs= best_epoch1+20, batch_size=batch_size, gpuid = gpuid, 
                                                   monitor='val_loss', patience=1000, verbose=0 ) #
                clf.fit(trainX, trainY, testX, testY)  #,  
                best_epoch = clf._performance.best_epoch + 1
                best_loss = round(clf._performance.best, 3)
                res = {'best_loss':best_loss, 'best_epoch':best_epoch,'scale_method': scale_method, 'fold_num':fold_num}
                print(res)
                all_res.append(res)
    df = pd.DataFrame(all_res)
    best_scale_method = df.groupby('scale_method')['best_loss'].mean().idxmin()
    best_epoch2 = df[df.scale_method == best_scale_method].best_epoch.median()
    best_epochs = int((best_epoch1+best_epoch2)/2)
    
    print('\n\nbest fill value: %s, best_scale_method:%s, best avg. best_epochs: %s' % (best_fill, best_scale_method, best_epochs))
    return best_fill, best_scale_method, best_epochs


def get_best_channel_number(dfx, dfy, best_fill = 1e-2, best_scale_method = 'minmax', best_epochs = 30, gpuid = 6):

    channel_numbers = [1, 5, 9, 13, 17, 21]
    all_res = []
    if len(dfx) < 500:
        batch_size = 2
    else:
        batch_size = 8   
    # fill values
    dfx_new = pd.DataFrame(np.log(dfx + best_fill))
    mp = AggMap(dfx_new)
        
    # get best channel number
    for cluster_channels in channel_numbers:
        mp = mp.fit(cluster_channels = cluster_channels, verbose = 0, var_thr = 0)            
        X = mp.batch_transform(dfx_new.values, scale_method = best_scale_method)
        Y = dfy.values
        outer = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 123)
        outer_idx = outer.split(range(len(dfy)), dfy.idxmax(axis=1))
        for i, idx in enumerate(list(outer_idx)):
            if i in [1, 3, 5, 7, 9]:        
                fold_num = "fold_%s" % str(i).zfill(2) 
                train_idx, test_idx = idx
                testY = Y[test_idx]
                testX = X[test_idx]
                trainX = X[train_idx]
                trainY = Y[train_idx]
                #print("\n %s: input train and test X shape is %s, %s " % (fold_num, trainX.shape,  testX.shape))
                clf = AggModel.MultiClassEstimator(epochs=best_epochs, batch_size=batch_size, gpuid = gpuid, 
                                                   monitor='val_loss', patience=1000, verbose=0) #
                clf.fit(trainX, trainY, testX, testY)  #,  
                best_epoch = clf._performance.best_epoch + 1
                best_loss = round(clf._performance.best, 3)
                res = {'best_loss':best_loss, 'best_epoch':best_epoch,'cluster_channels': cluster_channels, 'fold_num':fold_num,}
                print(res)
                all_res.append(res)

    df = pd.DataFrame(all_res)
    best_channel_number = df.groupby(['cluster_channels'])['best_loss'].mean().idxmin()
    best_epochs = df[df.cluster_channels == best_channel_number].best_epoch.median()
    best_epochs = int(best_epochs) + 10 
    print('\n\nbest channel number: %s' % best_channel_number)
    return best_channel_number, batch_size, best_epochs


def finetune_HPs(dfx, dfy,  gpuid = 6):
    
    best_fill, best_scale_method, best_epochs = get_best_fill_scale(dfx, dfy, gpuid = gpuid)
    best_channel_number, batch_size, best_epochs_update = get_best_channel_number(dfx, dfy, 
                                                              best_fill = best_fill, 
                                                              best_scale_method = best_scale_method, 
                                                              best_epochs = best_epochs,
                                                              gpuid = gpuid)
    
    return best_fill, best_scale_method, best_channel_number, best_epochs_update, batch_size