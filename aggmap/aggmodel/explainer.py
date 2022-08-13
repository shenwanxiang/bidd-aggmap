# -*- coding: utf-8 -*-
"""
Created on Fri Sep. 17 17:10:53 2021

@author: wanxiang.shen@u.nus.edu
"""


import numpy as np
import pandas as pd

from tqdm import tqdm
from copy import copy
import shap


from sklearn.metrics import mean_squared_error, log_loss
from sklearn.preprocessing import StandardScaler

from aggmap.utils.matrixopt import conv2
from aggmap.utils.logtools import print_warn, print_info



class shapley_explainer:
    """Kernel Shap based model explaination, the limiations can be found in this paper:https://christophm.github.io/interpretable-ml-book/shapley.html#disadvantages-16 <Problems with Shapley-value-based explanations as feature importance measures>. The SHAP values do not identify causality Global mean absolute Deep SHAP feature importance is the average impact on model output magnitude.
    
  
    Parameters
    ----------
    estimator:
        model with a predict or predict_probe method
    mp:
        aggmap object
    backgroud: string or int
        {'min', 'all', int}.
        if min, then use the min value as the backgroud data (equals to 1 sample)
        if int, then sample the K samples as the backgroud data
        if 'all' use all of the train data as the backgroud data for shap,
    k_means_sampling: bool,
        whether use the k-mean to sample the backgroud values or not
    link : 
        {"identity", "logit"}. A generalized linear model link to connect the feature importance values to the model output. 
        Since the feature importance values, phi, sum up to the model output, it often makes sense to connect them 
        to the output with a link function where link(output) = sum(phi). 
        If the model output is a probability then the LogitLink link function makes the feature importance values have log-odds units.
    args: 
        Other parameters for shap.KernelExplainer.
        
        
    
    Examples
    --------
    >>> import seaborn as sns
    >>> from aggmap.aggmodel.explainer import shapley_explainer
    >>> ## shapley explainer
    >>> shap_explainer = shapley_explainer(estimator, mp)
    >>> global_imp_shap = shap_explainer.global_explain(clf.X_)
    >>> local_imp_shap = shap_explainer.local_explain(clf.X_[[0]])
    >>> ## S-map of shapley explainer
    >>> sns.heatmap(local_imp_shap.shapley_importance_class_1.values.reshape(mp.fmap_shape), 
    >>> cmap = 'rainbow')
    >>> ## shapley plot
    >>> shap.summary_plot(shap_explainer.shap_values, 
    >>> feature_names = shap_explainer.feature_names) # #global  plot_type='bar
    >>> shap.initjs()
    >>> shap.force_plot(shap_explainer.explainer.expected_value[1], 
    >>> shap_explainer.shap_values[1], feature_names = shap_explainer.feature_names)

    """

    def __init__(self, estimator, mp, backgroud = 'min', k_means_sampling = True, link='identity', **args):
        '''
        
        Parameters
        ----------
        estimator:
            model with a predict or predict_probe method
        mp:
            aggmap object
        backgroud: string or int
            {'min', 'all', int}.
            if min, then use the min value as the backgroud data (equals to 1 sample)
            if int, then sample the K samples as the backgroud data
            if 'all' use all of the train data as the backgroud data for shap,
        k_means_sampling: bool,
            whether use the k-mean to sample the backgroud values or not
        link : 
            {"identity", "logit"}. A generalized linear model link to connect the feature importance values to the model output. 
            Since the feature importance values, phi, sum up to the model output, it often makes sense to connect them 
            to the output with a link function where link(output) = sum(phi). 
            If the model output is a probability then the LogitLink link function makes the feature importance values have log-odds units.
        args: 
            Other parameters for shap.KernelExplainer
        '''
        self.estimator = estimator
        self.mp = mp
        self.link = link
        self.backgroud = backgroud
        self.k_means_sampling = k_means_sampling
        
        train_features = self.covert_mpX_to_shapely_df(self.estimator.X_)
        
        if type(backgroud) == int:
            if self.k_means_sampling:
                self.backgroud_data =  shap.kmeans(train_features, backgroud)
            else:
                self.backgroud_data =  shap.sample(train_features, backgroud)
            
        else:
            if backgroud == 'min':
                self.backgroud_data =  train_features.min().to_frame().T.values
            else:
                self.backgroud_data =  train_features
        self.explainer = shap.KernelExplainer(self._predict_warpper, self.backgroud_data, link=self.link, **args)
        self.feature_names = train_features.columns.tolist() # mp.alist

        
    def _predict_warpper(self, X):
        X_new = self.mp.batch_transform(X, scale=False)
        if self.estimator.name == 'AggMap Regression Estimator': # case regression task
            predict_results = self.estimator.predict(X_new)
        else:
            predict_results = self.estimator.predict_proba(X_new)
        return predict_results
    
    def get_shap_values(self, X, nsamples = 'auto', **args):
        df_to_explain = self.covert_mpX_to_shapely_df(X)
        shap_values = self.explainer.shap_values(df_to_explain, nsamples=nsamples, **args)
        all_imps = []
        for i, data in enumerate(shap_values):
            name = 'shapley_importance_class_' + str(i) 
            imp = abs(pd.DataFrame(data, columns = self.feature_names)).mean().to_frame(name = name)
            all_imps.append(imp)

        df_reshape = self.mp.df_grid_reshape.set_index('v')
        df_reshape.index = self.mp.feature_names_reshape
        df_imp = df_reshape.join(pd.concat(all_imps, axis=1)).fillna(0)
        self.df_imp = df_imp
        self.shap_values = shap_values
        return shap_values
    
    
    def local_explain(self, X=None, idx=0, nsamples = 'auto', **args):
        
        '''
        Explaination of one sample only:
        
        Parameters
        ----------
        X: None or 4D array, where the shape is (n, w, h, c)
           the 4D array of AggMap multi-channel fmaps.
           Noted if X is None, then use the estimator.X_[[idx]] instead, namely explain the first sample if idx=0
        nsamples: {'auto', int}
            Number of times to re-evaluate the model when explaining each prediction. 
            More samples lead to lower variance estimates of the SHAP values. The “auto” setting uses nsamples = 2 * X.shape[1] + 2048
        args: other parameters in the shape_values method of shap.KernelExplainer 
        '''
        if X is None:
            print_info('Explaining the first sample only')
            X = self.clf.X_[[idx]]
        assert len(X.shape) == 4, 'input X mush a 4D array: (1, w, h, c)'
        assert len(X) == 1,  'Input X must has one sample only, but got %s' % len(X)
        
        shap_values = self.get_shap_values(X, nsamples = nsamples, **args)
        self.shap_values = shap_values
        return self.df_imp
    
    
    def global_explain(self, X=None, nsamples = 'auto', **args):
        '''
        Explaination of many samples.
        
        Parameters
        ----------
        X: None or 4D array, where the shape is (n, w, h, c)
           the 4D array of AggMap multi-channel fmaps.
           Noted that if X is None, then use the estimator.X_ instead, namely explain the training set of the estimator
        nsamples: {'auto', int}
            Number of times to re-evaluate the model when explaining each prediction. 
            More samples lead to lower variance estimates of the SHAP values. The “auto” setting uses nsamples = 2 * X.shape[1] + 2048
        args: other parameters in the shape_values method of shap.KernelExplainer 
        '''
        if X is None:
            X = self.clf.X_
            print_info('Explaining the whole samples of the training Set')
        assert len(X.shape) == 4, 'input X mush a 4D array: (n, w, h, c)'
        
        shap_values = self.get_shap_values(X, nsamples = nsamples, **args)
        self.shap_values = shap_values
        return self.df_imp

    
    def _covert_x_2D(self, X, feature_names):
        n, w,h, c = X.shape
        assert len(feature_names) == w*h, 'length of feature_names should be w*h of X.shape (n, w, h,c)'
        X_2D = X.sum(axis=-1).reshape(n, w*h)
        return pd.DataFrame(X_2D, columns = feature_names)


    def covert_mpX_to_shapely_df(self, X):
        dfx_stack_reshape = self._covert_x_2D(X, feature_names = self.mp.feature_names_reshape)
        shapely_df = pd.DataFrame(index=self.mp.alist).join(dfx_stack_reshape.T).T
        shapely_df = shapely_df.fillna(0)
        return shapely_df

    
    

class simply_explainer:
    
    """Simply-explainer for model explaination.

    Parameters
    ----------
    estimator:
        model with a predict or predict_probe method
    mp:
        aggmap object
    backgroud: 
        {'min', 'zeros'}, if 'zero' use all zeros as the backgroud data, if "min" use the min value of a vector of the training set
    apply_logrithm: bool, default: False
        apply the logirthm to the feature importance score
    apply_smoothing: bool, default: False
        apply the gaussian smoothing on the feature importance score (s-map )
    kernel_size:
        the kernel size for the smoothing
    sigma:
        the sigma for the smoothing.
        
    
    
            
    Examples
    --------
    >>> import seaborn as sns
    >>> from aggmap.aggmodel.explainer import simply_explainer
    >>> simp_explainer = simply_explainer(estimator, mp)
    >>> global_imp_simp = simp_explainer.global_explain(clf.X_, clf.y_)
    >>> local_imp_simp = simp_explainer.local_explain(clf.X_[[0]], clf.y_[[0]])    
    >>> ## S-map of simply explainer
    >>> sns.heatmap(local_imp_simp.simply_importance.values.reshape(mp.fmap_shape), 
    >>> cmap = 'rainbow')
    
    """
    
    def __init__(self, 
                 estimator, 
                 mp, 
                 backgroud = 'min', 
                 apply_logrithm = False,
                 apply_smoothing = False, 
                 kernel_size = 5, 
                 sigma = 1.
                ):
        '''
        Simply-explainer for model explaination.
        
        Parameters
        ----------
        estimator:
            model with a predict or predict_probe method
        mp:
            aggmap object
        backgroud: 
            {'min', 'zeros'}, if 'zero' use all zeros as the backgroud data, if "min" use the min value of a vector of the training set
        apply_logrithm: bool, default: False
            apply the logirthm to the feature importance score
        apply_smoothing: bool, default: False
            apply the gaussian smoothing on the feature importance score (s-map )
        kernel_size:
            the kernel size for the smoothing
        sigma:
            the sigma for the smoothing.
        '''
        self.estimator = estimator
        self.mp = mp
        self.apply_logrithm = apply_logrithm
        self.apply_smoothing = apply_smoothing
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.backgroud = backgroud
        if backgroud == 'min':
            self.backgroud_data =  mp.transform_mpX_to_df(self.estimator.X_).min().values
        else:
            self.backgroud_data =  np.zeros(shape=(len(mp.df_grid_reshape),))

        self.scaler = StandardScaler()

        df_grid = mp.df_grid_reshape.set_index('v')
        df_grid.index = self.mp.feature_names_reshape
        self.df_grid = df_grid
        
        if self.estimator.name == 'AggMap Regression Estimator':
            self._f = mean_squared_error
        else:
            self._f = log_loss
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _islice(self, lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]    
    
    
    def global_explain(self, X=None, y=None):
        '''
        Explaination of many samples.
        
        Parameters
        ----------
        X: None or 4D array, where the shape is (n, w, h, c)
           the 4D array of AggMap multi-channel fmaps
        y: None or 4D array, where the shape is (n, class_num)
           the True label
        Noted that if X and y are None, then use the estimator.X_ and estimator.y_ instead, namely explain the training set of the estimator
        '''
        
        if X is None:
            X = self.estimator.X_
            y = self.estimator.y_
            print_info('Explaining the whole samples of the training Set')
        
        assert len(X.shape) == 4, 'input X mush a 4D array: (n, w, h, c)'
        N, W, H, C = X.shape
        
        dfY = pd.DataFrame(y)
        Y_true = y
        Y_prob = self.estimator._model.predict(X)
        
        T = len(self.df_grid)
        nX = 20 # 10 arrX to predict

        if self.estimator.name == 'AggMap MultiLabels Estimator':
            Y_prob = self._sigmoid(Y_prob)
        final_res = {}
        for k, col in enumerate(dfY.columns):
            print_info('calculating feature importance for class %s ...' % col)
            results = []
            loss = self._f(Y_true[:, k].tolist(), Y_prob[:, k].tolist())
            
            tmp_X = []
            flag = 0
            for i in tqdm(range(T), ascii= True):
                ts = self.df_grid.iloc[i]
                y = ts.y
                x = ts.x
                ## step 1: make permutaions
                X1 = np.array(X)
                #x_min = X[:, y, x,:].min()
                vmin = self.backgroud_data[i]
                X1[:, y, x,:] = np.full(X1[:, y, x,:].shape, fill_value = vmin)
                tmp_X.append(X1)

                if (flag == nX) | (i == T-1):
                    X2p = np.concatenate(tmp_X)
                    ## step 2: make predictions
                    Y_pred_prob = self.estimator._model.predict(X2p) #predict ont by one is not efficiency
                    if self.estimator.name == 'AggMap MultiLabels Estimator':
                        Y_pred_prob = self._sigmoid(Y_pred_prob)    

                    ## step 3: calculate changes
                    for Y_pred in self._islice(Y_pred_prob, N):
                        mut_loss = self._f(Y_true[:, k].tolist(), Y_pred[:, k].tolist()) 
                        res =  mut_loss - loss # if res > 0, important, othervise, not important
                        results.append(res)
                    flag = 0
                    tmp_X = []
                flag += 1

            ## step 4:apply scaling or smothing 
            s = pd.DataFrame(results).values
            if self.apply_logrithm:
                s = np.log(s)
            smin = np.nanmin(s[s != -np.inf])
            smax = np.nanmax(s[s != np.inf])
            s = np.nan_to_num(s, nan=smin, posinf=smax, neginf=smin) #fillna with smin
            a = self.scaler.fit_transform(s)
            a = a.reshape(*self.mp.fmap_shape)
            if self.apply_smoothing:
                covda = conv2(a, kernel_size=self.kernel_size, sigma=self.sigma)
                results = covda.reshape(-1,).tolist()
            else:
                results = a.reshape(-1,).tolist()
            final_res.update({col:results})

        df = pd.DataFrame(final_res, index = self.mp.feature_names_reshape)
        df.columns = df.columns.astype(str)
        df.columns = 'simply_importance_class_' + df.columns
        df = self.df_grid.join(df)
        return df


    def local_explain(self, X=None, y=None, idx=0):
        '''
        Explaination of one sample only.
        
        Parameters
        ----------
        X: None or 4D array, where the shape is (1, w, h, c)
        y: the True label, None or 4D array, where the shape is (1, class_num).
        idx: int, 
             index of the sample to interpret
             Noted that if X and y are None, then use the estimator.X_[[idx]] and estimator.y_[[idx]] instead, namely explain the first sample if idx=0.
             
        Return
        ----------
            Feature importance of the current class
            

        '''
        
        if X is None:
            X = self.estimator.X_[[idx]]
            y = self.estimator.y_[[idx]]
            print_info('Explaining the one sample of the training Set')
        
        assert len(X.shape) == 4, 'input X mush a 4D array: (1, w, h, c)'
        assert (len(X) == 1) & (len(y) == 1), 'Input X, y must have one sample only, but got %s, %s' % (len(X), len(y))

        
        N, W, H, C = X.shape
        
        dfY = pd.DataFrame(y)
        Y_true = y
        Y_prob = self.estimator._model.predict(X)
        
        T = len(self.df_grid)
        nX = 20 # 10 arrX to predict

        if self.estimator.name == 'AggMap MultiLabels Estimator':
            Y_prob = self._sigmoid(Y_prob)

        results = []
        loss = self._f(Y_true.ravel().tolist(),  Y_prob.ravel().tolist())

        all_X1 = []
        for i in tqdm(range(T), ascii= True):
            ts = self.df_grid.iloc[i]
            y = ts.y
            x = ts.x
            X1 = np.array(X)
            vmin = self.backgroud_data[i]
            X1[:, y, x,:] = np.full(X1[:, y, x,:].shape, fill_value = vmin)
            all_X1.append(X1)

        all_X = np.concatenate(all_X1)
        all_Y_pred_prob = self.estimator._model.predict(all_X)

        for Y_pred_prob in all_Y_pred_prob:
            if self.estimator.name == 'AggMap MultiLabels Estimator':
                Y_pred_prob = self._sigmoid(Y_pred_prob)
            mut_loss = self._f(Y_true.ravel().tolist(), Y_pred_prob.ravel().tolist()) 
            res =  mut_loss - loss # if res > 0, important, othervise, not important
            results.append(res)

        ## apply smothing and scalings
        s = pd.DataFrame(results).values
        if self.apply_logrithm:
            s = np.log(s)
        smin = np.nanmin(s[s != -np.inf])
        smax = np.nanmax(s[s != np.inf])
        s = np.nan_to_num(s, nan=smin, posinf=smax, neginf=smin) #fillna with smin
        a = self.scaler.fit_transform(s)
        a = a.reshape(*self.mp.fmap_shape)
        if self.apply_smoothing:
            covda = conv2(a, kernel_size=self.kernel_size, sigma=self.sigma)
            results = covda.reshape(-1,).tolist()
        else:
            results = a.reshape(-1,).tolist()

        df = pd.DataFrame(results, 
                          index = self.mp.feature_names_reshape,
                          columns = ['simply_importance'])
        df = self.df_grid.join(df)
        return df

    

if __name__ == '__main__':
    '''
    Model explaination using two methods: simply explainer and shapley explainer
    '''
    
    import seaborn as sns
    
    ## simply explainer
    simp_explainer = simply_explainer(estimator, mp)
    global_imp_simp = simp_explainer.global_explain(clf.X_, clf.y_)
    local_imp_simp = simp_explainer.local_explain(clf.X_[[0]], clf.y_[[0]])    
    
    ## S-map of simply explainer
    sns.heatmap(local_imp_simp.simply_importance.values.reshape(mp.fmap_shape), cmap = 'rainbow')
    
    
    
    ## shapley explainer
    shap_explainer = shapley_explainer(estimator, mp)
    global_imp_shap = shap_explainer.global_explain(clf.X_)
    local_imp_shap = shap_explainer.local_explain(clf.X_[[0]])
    
    ## S-map of shapley explainer
    sns.heatmap(local_imp_shap.shapley_importance_class_1.values.reshape(mp.fmap_shape), cmap = 'rainbow')

    ## shapley plot
    shap.summary_plot(shap_explainer.shap_values, feature_names = shap_explainer.feature_names) # #global  plot_type='bar
    shap.initjs()
    shap.force_plot(shap_explainer.explainer.expected_value[1], shap_explainer.shap_values[1], feature_names = shap_explainer.feature_names)
