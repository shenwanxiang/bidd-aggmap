#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:29:36 2019

@author: wanxiang.shen@u.nus.edu

main aggmap code

"""
from aggmap.utils.logtools import print_info, print_warn, print_error
from aggmap.utils.matrixopt import Scatter2Grid, Scatter2Array, smartpadding 
from aggmap.utils import vismap, summary, calculator
from aggmap.utils.gen_nwk import mp2newick


from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding
        
from joblib import Parallel, delayed, load, dump
from scipy.spatial.distance import squareform, cdist, pdist
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
import matplotlib.pylab as plt
import seaborn as sns
from umap import UMAP
from tqdm import tqdm
from copy import copy, deepcopy
import pandas as pd
import numpy as np


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
    
    
class Random_2DEmbedding:
    
    def __init__(self, random_state=123, n_components=2):
        self.random_state=random_state
        self.n_components = n_components

    def fit(self, X):
        M, N = X.shape
        np.random.seed(self.random_state)
        rho = np.sqrt(np.random.uniform(0, 1, N))
        phi = np.random.uniform(0, 4*np.pi, N)
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        rd = pd.DataFrame([x,y]).T.sample(frac=1, random_state=123).reset_index(drop=True)
        self.embedding_ = rd.values
        return self
        
        
def _get_df_scatter(mp):
    xy = mp.embedded.embedding_
    colormaps = mp.colormaps
    df = pd.DataFrame(xy, columns = ['x', 'y'])
    bitsinfo = mp.bitsinfo.set_index('IDs')
    df = df.join(bitsinfo.loc[mp.flist].reset_index())
    df['colors'] = df['Subtypes'].map(colormaps)
    return df


def _get_df_grid(mp):

    p,q = mp._S.fmap_shape
    position = np.zeros(mp._S.fmap_shape, dtype='O').reshape(p*q,)
    position[mp._S.col_asses] = mp.flist
    position = position.reshape(p, q)
    if mp.fmap_shape != None:  
        m, n = mp.fmap_shape
        if (m > p) | (n > q):
            position = smartpadding(position, (m,n), constant_values=0)        
    M, N = position.shape
    
    x = []
    y = []
    for i in range(M):
        for j in range(N):
            x.append(j) #, position[j,i]
            y.append(i)
    v = position.reshape(M*N,)

    df = pd.DataFrame(list(zip(x,y, v)), columns = ['x', 'y', 'v'])

    bitsinfo = mp.bitsinfo
    subtypedict = bitsinfo.set_index('IDs')['Subtypes'].to_dict()
    subtypedict.update({0:'NaN'})
    df['Subtypes'] = df.v.map(subtypedict)
    df['colors'] = df['Subtypes'].map(mp.colormaps) 
    
    feature_list = df.v
    feature_names = []
    for i, j in feature_list.items():
        if j == 0:
            j = 'NaN-' + str(i)
        feature_names.append(j)
    df.v = feature_names
    return df


class AggMap(Base):
    
    """ The feature restructuring class AggMap
    
    
    Parameters
    ----------
    dfx: pandas DataFrame
        Input data frame. 
        
    metric: string,  default: 'correlation'
        measurement of feature distance, support {'cosine', 'correlation', 'euclidean', 'jaccard', 'hamming', 'dice'}
    
    info_distance: numpy array, defalt: None
        a vector-form distance vector of the feature points, shape should be: (n*(n-1)/2), where n is the number of the features. It can be useful when you have you own vector-form distance to pass
        
    by_scipy: bool, defalt: False.
        calculate the distance by using the scipy pdist fuction.
        It can bu useful when dfx.shape[1] > 20000, i.e., the number of features is very large
        Using pdist will increase the speed to calculate the distance, but may result a lower precision
    
    n_cpus: int, default: 16
        number of cpu cores to use to calculate the distance.        
    """
    
    def __init__(self, 
                 dfx,
                 metric = 'correlation',
                 by_scipy = False,
                 n_cpus = 16,
                 info_distance = None,
                ):
        
        assert type(dfx) == pd.core.frame.DataFrame, 'input dfx must be pandas DataFrame!'
        super().__init__()
        self.metric = metric
        self.by_scipy = by_scipy
        self.isfit = False
        self.alist = dfx.columns.tolist()
        self.ftype = 'feature points'
        self.cluster_flag = False
        m,n = dfx.shape
        info_distance_length = int(n*(n-1)/2)
        assert len(dfx.columns.unique()) == n, 'the column names of dataframe must be unique !'        
        
        ## calculating distance
        if np.array(info_distance).any():
            assert len(info_distance) == info_distance_length, 'shape of info_distance must be (%s,)' % info_distance_length
            print_info('Skipping the distance calculation, using the customized vector-form distance...')
            self.info_distance = np.array(info_distance)
            self.metric = 'precomputed'
        else:
            print_info('Calculating distance ...')
            
            if self.by_scipy:
                D = pdist(dfx.T, metric=metric)
                D = np.nan_to_num(D,copy=False)
                self.info_distance = D.clip(0, np.inf)
            else:
                D = calculator.pairwise_distance(dfx.values, n_cpus=n_cpus, method=metric)
                D = np.nan_to_num(D,copy=False)
                D_ = squareform(D)
                self.info_distance = D_.clip(0, np.inf)
            
        ## statistic info
        S = summary.Summary(n_jobs = 10)
        res= []
        for i in tqdm(range(dfx.shape[1]), ascii=True):
            r = S._statistics_one(dfx.values, i)
            res.append(r)
        dfs = pd.DataFrame(res, index = self.alist)
        self.info_scale = dfs
        
        
        
    def _fit_embedding(self, 
                       dist_matrix,
                       emb_method = 'umap', 
                       n_components = 2,
                       random_state = 32,  
                       verbose = 2,
                       n_neighbors = 15,
                       min_dist = 0.1,
                       a = 1.576943460405378,
                       b = 0.8950608781227859,
                       **kwargs):
        
        """
        parameters
        -----------------
        dist_matrix: distance matrix to fit
        emb_method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        kwargs: the extra parameters for the conresponding algorithm
        """
        

        if 'metric' in kwargs.keys():
            metric = kwargs.get('metric')
            kwargs.pop('metric')
            
        else:
            metric = 'precomputed'

        if emb_method == 'tsne':
            embedded = TSNE(n_components=n_components, 
                            random_state=random_state,
                            metric = metric,
                            verbose = verbose,
                            **kwargs)
            embedded = embedded.fit(dist_matrix)   
            
        elif emb_method == 'umap':
            embedded = UMAP(n_components = n_components, 
                            n_neighbors = n_neighbors,
                            min_dist = min_dist,
                            a = a,
                            b = b,
                            verbose = verbose,
                            random_state=random_state, 
                            metric = metric, **kwargs)
            embedded = embedded.fit(dist_matrix)   
            
        elif emb_method =='mds':
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
        
        elif emb_method == 'random':
            embedded = Random_2DEmbedding(random_state=random_state, 
                                          n_components=n_components)
            embedded = embedded.fit(dist_matrix)   
            
        elif emb_method == 'isomap':
            embedded = Isomap(n_neighbors = n_neighbors,
                              n_components=n_components, 
                              metric = metric,
                              **kwargs)
            embedded = embedded.fit(dist_matrix)   
            
        elif emb_method == 'lle':
            embedded = LocallyLinearEmbedding(random_state=random_state, 
                                              n_neighbors = n_neighbors,
                                              n_components=n_components, 
                                              **kwargs)
            embedded = embedded.fit(dist_matrix)   
            
        elif emb_method == 'se':
            embedded = SpectralEmbedding(random_state=random_state, 
                                          n_neighbors = n_neighbors,
                                          n_components=n_components, 
                                          affinity = metric,
                                          **kwargs)
            affinity_matrix = np.exp(-dist_matrix**2)  #make more uniformly embedding  
            
            embedded = embedded.fit(affinity_matrix)    
    
        return embedded


    
    def fit(self, 
            feature_group_list = [],
            cluster_channels = 5,
            var_thr = -1, 
            split_channels = True, 
            fmap_shape = None, 
            emb_method = 'umap', 
            min_dist = 0.1, 
            n_neighbors = 15,
            a = 1.576943460405378,
            b = 0.8950608781227859,
            verbose = 2, 
            random_state = 32,
            group_color_dict  = {},
            lnk_method = 'complete',
            **kwargs): 
        """
        parameters
        -----------------
        feature_group_list: list of the group name for each feature point
        cluster_channels: int, number of the channels(clusters) if feature_group_list is empty
        var_thr: float, defalt is -1, meaning that feature will be included only if the conresponding variance larger than this value. Since some of the feature has pretty low variances, we can remove them by increasing this threshold
        split_channels: bool, if True, outputs will split into various channels using the types of feature
        fmap_shape: None or tuple, size of molmap, if None, the size of feature map will be calculated automatically
        emb_method: {'umap', 'tsne', 'mds', 'isomap', 'random', 'lle', 'se'}, algorithm to embedd high-D to 2D
        min_dist: float, UMAP parameters for the effective minimum distance between embedded points.
        n_neighbors: init, UMAP parameters of controlling the embedding. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved.
        a: float, UMAP parameters of controlling the embedding. If None, it will automatically be determined by ``min_dist`` and ``spread``.
        b: float, UMAP parameters of controlling the embedding. If None, it will automatically be determined by ``min_dist`` and ``spread``.
        group_color_dict: dict of the group colors, keys are the group names, values are the colors
        lnk_method: {'complete', 'average', 'single', 'weighted', 'centroid'}, linkage method
        kwargs: the extra parameters for the conresponding embedding method
        """
            
        if 'n_components' in kwargs.keys():
            kwargs.pop('n_components')
            
            
        ## embedding  into a 2d 
        assert emb_method in ['tsne', 'umap', 'mds', 'isomap', 'random', 'lle', 'se'], 'No Such Method Supported: %s' % emb_method
        
        self.feature_group_list = feature_group_list
        self.var_thr = var_thr
        self.split_channels = split_channels
        self.fmap_shape = fmap_shape
        self.emb_method = emb_method
        self.lnk_method = lnk_method
        self.random_state = random_state
        
        if fmap_shape != None:
            assert len(fmap_shape) == 2, "fmap_shape must be a tuple with two elements!"
        
        # flist and distance
        flist = self.info_scale[self.info_scale['var'] > self.var_thr].index.tolist()
        
        dfd = pd.DataFrame(squareform(self.info_distance),
                           index=self.alist,
                           columns=self.alist)
        dist_matrix = dfd.loc[flist][flist]
        self.flist = flist
        
        self.x_mean = self.info_scale['mean'].values
        self.x_std =  self.info_scale['std'].values
        
        self.x_min = self.info_scale['min'].values
        self.x_max = self.info_scale['max'].values
     
        #bitsinfo
        dfb = pd.DataFrame(self.flist, columns = ['IDs'])
        if feature_group_list != []:
            
            self.cluster_flag = False
            
            assert len(feature_group_list) == len(self.alist), "the length of the input group list is not equal to length of the feature list"
            self.cluster_channels = len(set(feature_group_list))
            self.feature_group_list_ = feature_group_list
            dfb['Subtypes'] = dfb['IDs'].map(pd.Series(feature_group_list, index = self.alist))
            
            if set(feature_group_list).issubset(set(group_color_dict.keys())):
                self.group_color_dict = group_color_dict
                dfb['colors'] = dfb['Subtypes'].map(group_color_dict)
            else:
                unique_types = dfb['Subtypes'].unique()
                color_list = sns.color_palette("hsv", len(unique_types)).as_hex()
                group_color_dict = dict(zip(unique_types, color_list))
                dfb['colors'] = dfb['Subtypes'].map(group_color_dict)
                self.group_color_dict = group_color_dict
        else:
            self.cluster_channels = cluster_channels
            print_info('applying hierarchical clustering to obtain group information ...')
            self.cluster_flag = True
            
            Z = linkage(squareform(dist_matrix.values),  lnk_method)
            labels = fcluster(Z, cluster_channels, criterion='maxclust')
            
            feature_group_list_ = ['cluster_%s' % str(i).zfill(2) for i in labels]
            dfb['Subtypes'] = feature_group_list_
            dfb = dfb.sort_values('Subtypes')
            unique_types = dfb['Subtypes'].unique()
            
            if not set(unique_types).issubset(set(group_color_dict.keys())):
                color_list = sns.color_palette("hsv", len(unique_types)).as_hex()
                group_color_dict = dict(zip(unique_types, color_list))
            
            dfb['colors'] = dfb['Subtypes'].map(group_color_dict)
            self.group_color_dict = group_color_dict           
            self.Z = Z
            self.feature_group_list_ = feature_group_list_
            

        self.bitsinfo = dfb
        colormaps = dfb.set_index('Subtypes')['colors'].to_dict()
        colormaps.update({'NaN': '#000000'})
        self.colormaps = colormaps
        self._S = Scatter2Grid()

        ## 2d embedding first
        embedded = self._fit_embedding(dist_matrix,
                                       emb_method = emb_method,
                                       n_neighbors = n_neighbors,
                                       random_state = random_state,
                                       min_dist = min_dist, 
                                       a = a,
                                       b = b,
                                       verbose = verbose,
                                       n_components = 2, **kwargs)
        
        self.embedded = embedded 
        
        df = pd.DataFrame(embedded.embedding_, index = self.flist,columns=['x', 'y'])
        typemap = self.bitsinfo.set_index('IDs')
        df = df.join(typemap)
        df['Channels'] = df['Subtypes']
        self.df_embedding = df
      
        ## linear assignment algorithm 
        print_info('Applying grid assignment of feature points, this may take several minutes(1~30 min)')
        self._S.fit(self.df_embedding, self.split_channels, channel_col = 'Channels')
        print_info('Finished')
        
        ## fit flag
        self.isfit = True
        if self.fmap_shape == None:
            self.fmap_shape = self._S.fmap_shape        
        else:
            m, n = self.fmap_shape
            p, q = self._S.fmap_shape
            assert (m >= p) & (n >=q), "fmap_shape's width must >= %s, height >= %s " % (p, q)

    
        self.df_scatter = _get_df_scatter(self)
        self.df_grid = _get_df_grid(self)
        self.df_grid_reshape = _get_df_grid(self)
        self.feature_names_reshape = self.df_grid.v.tolist()
        return self
        
    
    def refit_c(self, cluster_channels = 10, lnk_method = 'complete', group_color_dict = {}):
        """
        re-fit the aggmap object to update the number of channels
        
        parameters
        --------------------
        cluster_channels: int, number of the channels(clusters)
        group_color_dict: dict of the group colors, keys are the group names, values are the colors
        lnk_method: {'complete', 'average', 'single', 'weighted', 'centroid'}, linkage method
        """
        
        if not self.isfit:
            print_error('please fit first!')
            return
            

        self.split_channels = True
        self.lnk_method = lnk_method
        self.cluster_channels = cluster_channels
        
        dfd = pd.DataFrame(squareform(self.info_distance),
                           index=self.alist,
                           columns=self.alist)
        dist_matrix = dfd.loc[self.flist][self.flist]

        dfb = pd.DataFrame(self.flist, columns = ['IDs'])
        print_info('applying hierarchical clustering to obtain group information ...')
        self.cluster_flag = True

        Z = linkage(squareform(dist_matrix.values),  self.lnk_method)
        labels = fcluster(Z, self.cluster_channels, criterion='maxclust')

        feature_group_list_ = ['cluster_%s' % str(i).zfill(2) for i in labels]
        dfb['Subtypes'] = feature_group_list_
        dfb = dfb.sort_values('Subtypes')
        unique_types = dfb['Subtypes'].unique()

        if not set(unique_types).issubset(set(group_color_dict.keys())):
            color_list = sns.color_palette("hsv", len(unique_types)).as_hex()
            group_color_dict = dict(zip(unique_types, color_list))

        dfb['colors'] = dfb['Subtypes'].map(group_color_dict)
        self.group_color_dict = group_color_dict           
        self.Z = Z
        self.feature_group_list_ = feature_group_list_

        # update self.bitsinfo
        self.bitsinfo = dfb
        colormaps = dfb.set_index('Subtypes')['colors'].to_dict()
        colormaps.update({'NaN': '#000000'})
        self.colormaps = colormaps

        # update self.df_embedding
        df = pd.DataFrame(self.embedded.embedding_, index = self.flist,columns=['x', 'y'])
        typemap = self.bitsinfo.set_index('IDs')
        df = df.join(typemap)
        df['Channels'] = df['Subtypes']
        self.df_embedding = df

        ## linear assignment not performed, only refit the channel number 
        print_info('skipping grid assignment of feature points, fitting to target channel number')
        self._S.refit_c(self.df_embedding)
        print_info('Finished')

        if self.fmap_shape == None:
            self.fmap_shape = self._S.fmap_shape        
        else:
            m, n = self.fmap_shape
            p, q = self._S.fmap_shape
            assert (m >= p) & (n >=q), "fmap_shape's width must >= %s, height >= %s " % (p, q)


        self.df_scatter = _get_df_scatter(self)
        self.df_grid = _get_df_grid(self)
        self.df_grid_reshape = _get_df_grid(self)
        self.feature_names_reshape = self.df_grid.v.tolist()
        return self
    
    def transform_mpX_to_df(self, X):
        '''
        input 4D X, output 2D dataframe
        '''
        n, w,h, c = X.shape
        X_2D = X.sum(axis=-1).reshape(n, w*h)
        return pd.DataFrame(X_2D, columns = self.feature_names_reshape)    

    
    def transform(self, 
                  arr_1d, 
                  scale = True, 
                  scale_method = 'minmax',
                  fillnan = 0):
    
    
        """
        parameters
        --------------------
        arr_1d: 1d numpy array feature points
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        fillnan: fill nan value, default: 0
        """
        
        if not self.isfit:
            print_error('please fit first!')
            return

        if scale:
            if scale_method == 'standard':
                arr_1d = self.StandardScaler(arr_1d, self.x_mean, self.x_std)
            else:
                arr_1d = self.MinMaxScaleClip(arr_1d, self.x_min, self.x_max)
        
        df = pd.DataFrame(arr_1d).T
        df.columns = self.alist

        df = df[self.flist]
        vector_1d = df.values[0] #shape = (N, )
        fmap = self._S.transform(vector_1d)  
        p, q, c = fmap.shape
        
        if self.fmap_shape != None:        
            m, n = self.fmap_shape
            if (m > p) | (n > q):
                fps = []
                for i in range(c):
                    fp = smartpadding(fmap[:,:,i], self.fmap_shape)
                    fps.append(fp)
                fmap = np.stack(fps, axis=-1)        

        return np.nan_to_num(fmap, nan = fillnan)   
    
    

    
    def batch_transform(self, 
                        array_2d, 
                        scale = True, 
                        scale_method = 'minmax',
                        n_jobs=4,
                        fillnan = 0):
    
        """
        parameters
        --------------------
        array_2d: 2D numpy array feature points, M(samples) x N(feature ponits)
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        n_jobs: number of parallel
        fillnan: fill nan value, default: 0
        """
        
        if not self.isfit:
            print_error('please fit first!')
            return
        
        assert type(array_2d) == np.ndarray, 'input must be numpy ndarray!' 
        assert array_2d.ndim == 2, 'input must be 2-D  numpy array!' 
        
        P = Parallel(n_jobs=n_jobs)
        res = P(delayed(self.transform)(arr_1d, 
                                        scale,
                                        scale_method,
                                        fillnan) for arr_1d in tqdm(array_2d, ascii=True)) 
        X = np.stack(res) 
        
        return X
    
    
    def plot_scatter(self, htmlpath='./', htmlname=None, radius = 2, enabled_data_labels = False):
        """Scatter plot, radius: the size of the scatter, must be int"""
        H_scatter = vismap.plot_scatter(self,  
                                        htmlpath=htmlpath, 
                                        htmlname=htmlname,
                                        radius = radius,
                                        enabled_data_labels = enabled_data_labels)
        return H_scatter   
        
        
    def plot_grid(self, htmlpath='./', htmlname=None, enabled_data_labels = False):
        """Grid plot"""
        
        H_grid = vismap.plot_grid(self,
                                  htmlpath=htmlpath, 
                                  htmlname=htmlname,
                                  enabled_data_labels = enabled_data_labels)
        return H_grid       
        
        
        
    def plot_tree(self, figsize=(16,8), add_leaf_labels = True, leaf_font_size = 18, leaf_rotation = 90):
        """Diagram tree plot"""
            
        fig = plt.figure(figsize=figsize)
        
        if self.cluster_flag:
            
            Z = self.Z
            

            D_leaf_colors = self.bitsinfo['colors'].to_dict() 
            link_cols = {}
            for i, i12 in enumerate(Z[:,:2].astype(int)):
                c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors[x] for x in i12)
                link_cols[i+1+len(Z)] = c1
            
            if add_leaf_labels:
                labels = self.alist
            else:
                labels = None
            P =dendrogram(Z, labels = labels, 
                          leaf_rotation = leaf_rotation, 
                          leaf_font_size = leaf_font_size, 
                          link_color_func=lambda x: link_cols[x])
        
        return fig
        
        
    def to_nwk_tree(self, treefile = 'mytree', leaf_names = None):
        '''
        convert mp object to newick tree and the data file to submit to itol sever
        '''
        return mp2newick(self, treefile = treefile, leaf_names=leaf_names)
        
        
    def copy(self):
        """copy self"""
        return deepcopy(self)
        
        
    def load(self, filename):
        """load self"""
        return self._load(filename)
    
    
    def save(self, filename):
        """save self"""
        return self._save(filename)
