#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:29:36 2019

@author: wanxiang.shen@u.nus.edu

main rfmap code

"""


from rfmap.utils.logtools import print_info, print_warn, print_error
from rfmap.utils.matrixopt import Scatter2Grid, Scatter2Array 
from rfmap.utils import vismap, summary, calculator


from sklearn.manifold import TSNE, MDS
from sklearn.utils import shuffle
from joblib import Parallel, delayed, load, dump
from scipy.spatial.distance import squareform
from umap import UMAP
from tqdm import tqdm
import pandas as pd
import numpy as np
import os



class Base:
    
    def __init__(self):
        pass
        
    def _save(self, filename):
        return dump(self, filename)
        
    def _load(self, filename):
        return load(filename)
 

    def MinMaxScaleClip(self, x, xmin, xmax):
        scaled = (x - xmin) / ((xmax - xmin) + 1e-8)
        return scaled

    def StandardScaler(self, x, xmean, xstd):
        return (x-xmean) / (xstd + 1e-8) 
    

    

class RFMAP(Base):
    
    def __init__(self, 
                 dfx,
                 metric = 'cosine' 
                ):
        
        """
        paramters
        -----------------
        dfx: pandas DataFrame
        metric: {'cosine', 'correlation', 'euclidean', 'jaccard', 'hamming', 'dice'}, default: 'cosine', measurement of feature distance

        
        """
        super().__init__()

        self.metric = metric
        self.isfit = False
        self.alist = dfx.columns.tolist()
        self.ftype = 'feature points'
        ## calculating distance
        print_info('Calculating distance ...')
        D = calculator.pairwise_distance(dfx.values, n_cpus=16, method=metric)
        D = np.nan_to_num(D,copy=False)
        self.info_distance = squareform(D)

        
        
        ## statistic info
        S = summary.Summary(n_jobs = 10)
        res= []
        for i in tqdm(range(dfx.shape[1])):
            r = S._statistics_one(dfx.values, i)
            res.append(r)
        dfs = pd.DataFrame(res, index = self.alist)
        self.info_scale = dfs
        
        
        
    def _fit_embedding(self, 
                        dist_matrix,
                        method = 'tsne',  
                        n_components = 2,
                        random_state = 1,  
                        verbose = 2,
                        n_neighbors = 30,
                        min_dist = 0.1,
                        **kwargs):
        
        """
        parameters
        -----------------
        dist_matrix: distance matrix to fit
        method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        kwargs: the extra parameters for the conresponding algorithm
        """

        if 'metric' in kwargs.keys():
            metric = kwargs.get('metric')
            kwargs.pop('metric')
            
        else:
            metric = 'precomputed'

        if method == 'tsne':
            embedded = TSNE(n_components=n_components, 
                            random_state=random_state,
                            metric = metric,
                            verbose = verbose,
                            **kwargs)
        elif method == 'umap':
            embedded = UMAP(n_components = n_components, 
                            n_neighbors = n_neighbors,
                            min_dist = min_dist,
                            verbose = verbose,
                            random_state=random_state, 
                            metric = metric, **kwargs)
            
        elif method =='mds':
            if 'metric' in kwargs.keys():
                kwargs.pop('metric')
            if 'dissimilarity' in kwargs.keys():
                dissimilarity = kwargs.get('dissimilarity')
                kwargs.pop('dissimilarity')
            else:
                dissimilarity = 'precomputed'
                
            embedded = MDS(metric = True, 
                           n_components= n_components,
                           verbose = verbose,
                           dissimilarity = dissimilarity, 
                           random_state = random_state, **kwargs)
        
        embedded = embedded.fit(dist_matrix)    
        
        return embedded
    
    
   
            

    def fit(self, 
            feature_group_list = [],
            group_color_dict  = {},
            var_thr = 1e-3, 
            split_channels = True, 
            fmap_type = 'grid',  
            fmap_shape = None, 
            emb_method = 'umap', 
            min_dist = 0.1, 
            n_neighbors = 15,
            verbose = 2, 
            random_state = 32, 
            **kwargs): 
        """
        parameters
        -----------------
        feature_group_list: list of the group name for each feature point
        group_color_dict: dict of the group colors, keys are the group names, values are the colors
        var_thr: float, defalt is 1e-3, meaning that feature will be included only if the conresponding variance larger than this value. Since some of the feature has pretty low variances, we can remove them by increasing this threshold
        split_channels: bool, if True, outputs will split into various channels using the types of feature
        fmap_type:{'scatter', 'grid'}, default: 'gird', if 'scatter', will return a scatter mol map without an assignment to a grid
        fmap_shape: None or tuple, size of molmap, only works when fmap_type is 'scatter', if None, the size of feature map will be calculated automatically
        emb_method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        kwargs: the extra parameters for the conresponding embedding method
        """
            
        if 'n_components' in kwargs.keys():
            kwargs.pop('n_components')
            
        #bitsinfo
        dfb = pd.DataFrame(self.alist, columns = ['IDs'])
        if feature_group_list != []:
            assert len(feature_group_list) == len(self.alist), "the length of the input group list is not equal to length of the feature list"
            dfb['Subtypes'] = feature_group_list
            if group_color_dict != {}:
                assert set(feature_group_list).issubset(set(group_color_dict.keys())), 'group_color_dict should contains all of the feature groups'
                dfb['colors'] = dfb['Subtypes'].map(group_color_dict)
            else:
                dfb['colors'] = '#ff6a00' 
        else:
            dfb['Subtypes'] = 'features'
            dfb['colors'] = '#ff6a00'
        self.bitsinfo = dfb
        colormaps = dfb.set_index('Subtypes')['colors'].to_dict()
        colormaps.update({'NaN': '#000000'})
        self.colormaps = colormaps
        
        ## embedding  into a 2d 
        assert emb_method in ['tsne', 'umap', 'mds'], 'No Such Method Supported: %s' % emb_method
        assert fmap_type in ['scatter', 'grid'], 'No Such Feature Map Type Supported: %s'   % fmap_type     
        self.var_thr = var_thr
        self.split_channels = split_channels
        self.fmap_type = fmap_type
        self.fmap_shape = fmap_shape
        self.emb_method = emb_method

        
        scale_info = self.info_scale[self.info_scale['var'] > self.var_thr]
        flist = scale_info.index.tolist()
        
        dfd = pd.DataFrame(squareform(self.info_distance),
                           index=self.alist,
                           columns=self.alist)

        dist_matrix = dfd.loc[flist][flist]
        
        self.flist = flist
        self.scale_info = scale_info
        

        if fmap_type == 'grid':
            S = Scatter2Grid()
        else:
            if fmap_shape == None:
                N = len(self.flist)
                l = np.int(np.sqrt(N))*2
                fmap_shape = (l, l)                
            S = Scatter2Array(fmap_shape)
        
        self._S = S

        ## 2d embedding first
        embedded = self._fit_embedding(dist_matrix,
                                       method = emb_method,
                                       n_neighbors = n_neighbors,
                                       random_state = random_state,
                                       min_dist = min_dist, 
                                       verbose = verbose,
                                       n_components = 2, **kwargs)
        
        self.embedded = embedded 
        
        df = pd.DataFrame(embedded.embedding_, index = self.flist,columns=['x', 'y'])
        typemap = self.bitsinfo.set_index('IDs')
        df = df.join(typemap)
        df['Channels'] = df['Subtypes']
        self.df_embedding = df
      
        if self.fmap_type == 'scatter':
            ## naive scatter algorithm
            print_info('Applying naive scatter feature map...')
            self._S.fit(self.df_embedding, self.split_channels, channel_col = 'Channels')
            print_info('Finished')
            
        else:
            ## linear assignment algorithm 
            print_info('Applying grid feature map(assignment), this may take several minutes(1~30 min)')
            self._S.fit(self.df_embedding, self.split_channels, channel_col = 'Channels')
            print_info('Finished')
        
        ## fit flag
        self.isfit = True
        self.fmap_shape = self._S.fmap_shape        
        
        

    def transform(self, 
                  arr_1d, 
                  scale = True, 
                  scale_method = 'minmax',):
    
    
        """
        parameters
        --------------------
        arr_1d: 1d numpy array feature points
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        """
        
        if not self.isfit:
            print_error('please fit first!')
            return


        df = pd.DataFrame(arr_1d).T
        df.columns = self.bitsinfo.IDs
        
        if scale:
            
            if scale_method == 'standard':
                df = self.StandardScaler(df,  
                                    self.scale_info['mean'],
                                    self.scale_info['std'])
            else:
                df = self.MinMaxScaleClip(df, 
                                     self.scale_info['min'], 
                                     self.scale_info['max'])
        
        df = df[self.flist]
        vector_1d = df.values[0] #shape = (N, )
        fmap = self._S.transform(vector_1d)       
        return np.nan_to_num(fmap)   

    
    def batch_transform(self, 
                        array_2d, 
                        scale = True, 
                        scale_method = 'minmax',
                        n_jobs=4):
    
        """
        parameters
        --------------------
        array_2d: 2D numpy array feature points, M(samples) x N(feature ponits)
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        n_jobs: number of parallel
        """
        
        assert array_2d.ndim == 2, 'input X must be 2-D array!' 
        
        P = Parallel(n_jobs=n_jobs)
        res = P(delayed(self.transform)(arr_1d, 
                                        scale,
                                        scale_method) for arr_1d in tqdm(array_2d, ascii=True)) 
        X = np.stack(res) 
        
        return X
    
    
    def plot_scatter(self, htmlpath='./', htmlname=None, radius = 2):
        """radius: the size of the scatter, must be int"""
        df_scatter, H_scatter = vismap.plot_scatter(self,  
                                htmlpath=htmlpath, 
                                htmlname=htmlname,
                                radius = radius)
        
        self.df_scatter = df_scatter
        return H_scatter   
        
        
    def plot_grid(self, htmlpath='./', htmlname=None):
        
        if self.fmap_type != 'grid':
            return
        
        df_grid, H_grid = vismap.plot_grid(self,  
                                htmlpath=htmlpath, 
                                htmlname=htmlname)
        
        self.df_grid = df_grid
        return H_grid       
        
        
    def load(self, filename):
        return self._load(filename)
    
    
    def save(self, filename):
        return self._save(filename)