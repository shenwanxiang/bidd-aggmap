# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 17:10:53 2020

@author: wanxiang.shen@u.nus.edu
"""

import warnings, os
warnings.filterwarnings("ignore")


import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import get_scorer, SCORERS

from aggmap import aggmodel

from joblib import dump, load
from  copy import copy
from tensorflow.keras.models import load_model as load_tf_model


def save_model(model, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print('saving model to %s' % model_path)
    model_new = copy(model)
    model_new._model.save(os.path.join(model_path, 'inner_model.h5'))
    model_new._model = None
    model_new._performance = None
    dump(model_new,  os.path.join(model_path, 'outer_model.est'))
    
    
def load_model(model_path, gpuid=0):
    '''
    gpuid: load model to specific gpu
    '''
    gpuid = str(gpuid)
    os.environ["CUDA_VISIBLE_DEVICES"]= gpuid
    model = load(os.path.join(model_path, 'outer_model.est'))
    model.gpuid = gpuid
    model._model = load_tf_model(os.path.join(model_path, 'inner_model.h5'))
    return model



class RegressionEstimator(BaseEstimator, RegressorMixin):
    """ An AggMap CNN Regression estimator (each sample belongs to only one class) 
    Parameters
    ----------
    epochs : int, default = 200
        A parameter used for training epochs. 
    conv1_kernel_size: int, default = 13
        A parameter used for the kernel size of first covolutional layers
    dense_layers: list, default = [128]
        A parameter used for the dense layers.  
    batch_size: int, default: 128
        A parameter used for the batch size.
    lr: float, default: 1e-4
        A parameter used for the learning rate.
    batch_norm: bool, default: False
        batch normalization after convolution layers.
    n_inception: int, default:2
        Number of the inception layers.
    dense_avf: str, default is 'relu'
        activation fuction in the dense layers.
    dropout: float, default: 0
        A parameter used for the dropout of the dense layers.
    monitor: str, default: 'val_loss'
        {'val_loss', 'val_r2'}, a monitor for model selection
    metric: str, default: 'r2'
        {'r2', 'rmse'},  a matric parameter
    patience: int, default = 10000, 
        A parameter used for early stopping
    gpuid: int, default=0,
        A parameter used for specific gpu card
    verbose: int, default = 0
    random_state, int, default: 32
    name: str 

    Examples
    --------
    >>> from aggmap import AggModel
    >>> clf = AggModel.RegressionEstimator()

    """
    
    def __init__(self, 
                 epochs = 200,  
                 conv1_kernel_size = 13,
                 dense_layers = [128],  
                 dense_avf = 'relu',
                 batch_size = 128,  
                 lr = 1e-4, 
                 batch_norm = False,
                 n_inception = 2,
                 dropout = 0.0,
                 monitor = 'val_loss', 
                 metric = 'r2',
                 patience = 10000,
                 verbose = 0, 
                 random_state = 32,
                 gpuid = 0,
                 name = "AggMap Regression Estimator"
                ):
        
        
        self.epochs = epochs
        self.dense_layers = dense_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.dense_avf = dense_avf
        self.batch_size = batch_size
        self.lr = lr
        self.batch_norm = batch_norm
        self.n_inception = n_inception
        self.dropout = dropout
        self.monitor = monitor
        self.metric = metric
        self.patience = patience
        
        self.gpuid = str(gpuid)
        os.environ["CUDA_VISIBLE_DEVICES"]= self.gpuid
        
        self.verbose = verbose
        self.random_state = random_state
        self.is_fit = False        
        self.name = name

        print(self.get_params())
        
        
    def get_params(self, deep=True):

        model_paras =  {"epochs": self.epochs, 
                        "lr":self.lr, 
                        "conv1_kernel_size": self.conv1_kernel_size,
                        "dense_layers": self.dense_layers, 
                        "dense_avf":self.dense_avf, 
                        "batch_size":self.batch_size, 
                        "dropout":self.dropout,
                        "batch_norm":self.batch_norm,
                        "n_inception":self.n_inception,
                        "monitor": self.monitor,
                        "patience":self.patience,
                        "random_state":self.random_state,
                        "verbose":self.verbose,
                        "name":self.name,
                        "gpuid": self.gpuid,
                       }

        return model_paras
    
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

        

    def fit(self, X, y,  X_valid = None, y_valid = None):

        # Check that X and y have correct shape
        
        if  X.ndim != 4:
            raise ValueError("Found array X with dim %d. %s expected == 4." % (X.ndim, self.name))
            
        if  y.ndim != 2:
            raise ValueError("Found array y with dim %d. %s expected == 2." % (y.ndim, self.name))    
    
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        
        if (X_valid is None) | (y_valid is None):
            X_valid = X
            y_valid = y
        
        np.random.seed(self.random_state)
        tf.compat.v1.set_random_seed(self.random_state)
        

        model = aggmodel.net.AggMapNet2(X.shape[1:],
                                       n_outputs = y.shape[-1], 
                                       conv1_kernel_size = self.conv1_kernel_size,
                                       batch_norm = self.batch_norm,
                                       n_inception = self.n_inception,
                                       dense_layers = self.dense_layers, 
                                       dense_avf = self.dense_avf, 
                                       dropout = self.dropout,
                                       last_avf = 'linear')  


        opt = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
        model.compile(optimizer = opt, loss = 'mse')
        performance = aggmodel.cbks.Reg_EarlyStoppingAndPerformance((X, y), 
                                                                    (X_valid, y_valid), 
                                                                    patience = self.patience, 
                                                                    criteria = self.monitor,
                                                                    verbose = self.verbose,)

        model.fit(X, y, 
                  batch_size=self.batch_size, 
                  epochs= self.epochs, verbose= 0, shuffle = True, 
                  validation_data = (X_valid, y_valid), 
                  callbacks=[performance]) 

        self._model = model
        self._performance = performance
        self.is_fit = True
        # Return the classifier
        return self


    
    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_w, n_features_h, n_features_c)
            Vector to be scored, where `n_samples` is the number of samples and

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        y_pred = self._model.predict(X)
        return y_pred
    


    def score(self, X, y, scoring = 'r2', sample_weight=None):
        """Returns the score using the `scoring` option on the given
        test data and labels.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        scoring: str, please refer to: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """
        assert scoring in SCORERS.keys(), 'scoring is not in %s' % SCORERS.keys()
        scoring = get_scorer(scoring)

        return scoring(self, X, y, sample_weight=sample_weight)  
    
    
    def plot_model(self, to_file='model.png', 
                   show_shapes=True, 
                   show_layer_names=True, 
                   rankdir='TB', 
                   expand_nested=False, 
                   dpi=96):
        if self.is_fit:
            tf.keras.utils.plot_model(self._model, 
                       to_file=to_file, 
                       show_shapes=show_shapes, 
                       show_layer_names=show_layer_names, 
                       rankdir=rankdir, 
                       expand_nested=expand_nested, 
                       dpi=dpi)
        else:
            print('Please fit first!')
    
    
    
    
class MultiClassEstimator(BaseEstimator, ClassifierMixin):

    """ An AggMap CNN MultiClass estimator (each sample belongs to only one class) 
    Parameters
    ----------
    epochs : int, default = 200
        A parameter used for training epochs. 
    conv1_kernel_size: int, default = 13
        A parameter used for the kernel size of first covolutional layers
    dense_layers: list, default = [128]
        A parameter used for the dense layers.  
    batch_size: int, default: 128
        A parameter used for the batch size.
    lr: float, default: 1e-4
        A parameter used for the learning rate.
    batch_norm: bool, default: False
        batch normalization after convolution layers.
    n_inception: int, default:2
        Number of the inception layers.
    dense_avf: str, default is 'relu'
        activation fuction in the dense layers.
    dropout: float, default: 0
        A parameter used for the dropout of the dense layers.
    monitor: str, default: 'val_loss'
        {'val_loss', 'val_auc'}, a monitor for model selection
    metric: str, default: 'ROC'
        {'ROC', 'ACC', 'PRC'},  a matric parameter
    patience: int, default = 10000, 
        A parameter used for early stopping
    gpuid: int, default=0,
        A parameter used for specific gpu card
    verbose: int, default = 0
    random_state, int, default: 32
    name: str 

    Examples
    --------
    >>> from aggmap import AggModel
    >>> clf = AggModel.MultiClassEstimator()
    """
    
    
    def __init__(self, 
                 epochs = 200,  
                 conv1_kernel_size = 13,
                 dense_layers = [128],  
                 dense_avf = 'relu',
                 batch_size = 128,  
                 lr = 1e-4, 
                 batch_norm = False,
                 n_inception = 2,                 
                 dropout = 0.0,
                 monitor = 'val_loss', 
                 metric = 'ROC',
                 patience = 10000,
                 verbose = 0, 
                 random_state = 32,
                 name = "AggMap MultiClass Estimator",
                 gpuid=0,
                ):
        
        
        self.epochs = epochs
        self.dense_layers = dense_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.dense_avf = dense_avf
        self.batch_size = batch_size
        self.lr = lr
        self.batch_norm = batch_norm
        self.n_inception = n_inception      
        self.dropout = dropout
        
        self.monitor = monitor
        self.metric = metric
        self.patience = patience
        
        self.gpuid = str(gpuid)
        os.environ["CUDA_VISIBLE_DEVICES"]= self.gpuid
        
        self.verbose = verbose
        self.random_state = random_state
        
        self.name = name
        self.is_fit = False        
        print(self.get_params())
        
        
    def get_params(self, deep=True):

        model_paras =  {"epochs": self.epochs, 
                        "lr":self.lr, 
                        "conv1_kernel_size": self.conv1_kernel_size,
                        "dense_layers": self.dense_layers, 
                        "dense_avf":self.dense_avf, 
                        "batch_size":self.batch_size, 
                        "dropout":self.dropout,
                        "batch_norm":self.batch_norm,
                        "n_inception":self.n_inception,                        
                        "monitor": self.monitor,
                        "patience":self.patience,
                        "random_state":self.random_state,
                        "verbose":self.verbose,
                        "name":self.name,
                        "gpuid": self.gpuid,
                       }

        return model_paras
    
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

        

    def fit(self, X, y,  
            X_valid = None,
            y_valid = None, 
            loss = 'categorical_crossentropy', 
            last_avf = 'softmax', 
            class_weight = None,
           ):

        # Check that X and y have correct shape
        
        if  X.ndim != 4:
            raise ValueError("Found array X with dim %d. %s expected == 4." % (X.ndim, self.name))
            
        if  y.ndim != 2:
            raise ValueError("Found array y with dim %d. %s expected == 2." % (y.ndim, self.name))    
    
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        
        if (X_valid is None) | (y_valid is None):
            
            X_valid = X
            y_valid = y
        
        np.random.seed(self.random_state)
        tf.compat.v1.set_random_seed(self.random_state)


        model = aggmodel.net.AggMapNet2(X.shape[1:],
                                        n_outputs = y.shape[-1], 
                                        conv1_kernel_size = self.conv1_kernel_size,
                                        batch_norm = self.batch_norm,
                                        n_inception = self.n_inception,
                                        dense_layers = self.dense_layers, 
                                        dense_avf = self.dense_avf, 
                                        dropout = self.dropout,
                                        last_avf = last_avf)


        opt = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
        model.compile(optimizer = opt, loss = loss, metrics = ['accuracy'])
        performance = aggmodel.cbks.CLA_EarlyStoppingAndPerformance((X, y), 
                                                                    (X_valid, y_valid), 
                                                                    patience = self.patience, 
                                                                    criteria = self.monitor,
                                                                    metric = self.metric,  
                                                                    last_avf= last_avf,
                                                                    verbose = 0,)

        history = model.fit(X, y, 
                  batch_size=self.batch_size, 
                  epochs= self.epochs, verbose= self.verbose, shuffle = True, 
                  validation_data = (X_valid, y_valid), class_weight = class_weight, 
                  callbacks=[performance]) 

        self._model = model
        self._performance = performance
        self.history = history
        self.is_fit = True        
        # Return the classifier
        return self



    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        if  X.ndim != 4:
            raise ValueError("Found array X with dim %d. %s expected == 4." % (X.ndim, self.name))
        y_prob = self._model.predict(X)
        return y_prob
    

    def predict(self, X):
        y_pred = np.round(self.predict_proba(X))
        return y_pred
    
    

    def score(self, X, y, scoring = 'accuracy', sample_weight=None):
        """Returns the score using the `scoring` option on the given
        test data and labels.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        scoring: str, please refer to: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """
        assert scoring in SCORERS.keys(), 'scoring is not in %s' % SCORERS.keys()
        scoring = get_scorer(scoring)

        return scoring(self, X, y, sample_weight=sample_weight)
    
    
    def plot_model(self, to_file='model.png', 
                   show_shapes=True, 
                   show_layer_names=True, 
                   rankdir='TB', 
                   expand_nested=False, 
                   dpi=96):
        if self.is_fit:
            tf.keras.utils.plot_model(self._model, 
                       to_file=to_file, 
                       show_shapes=show_shapes, 
                       show_layer_names=show_layer_names, 
                       rankdir=rankdir, 
                       expand_nested=expand_nested, 
                       dpi=dpi)
        else:
            print('Please fit first!')    
    

class MultiLabelEstimator(BaseEstimator, ClassifierMixin):


    """ An AggMap CNN MultiLabel estimator (each sample belongs to only one class) 
    Parameters
    ----------
    epochs : int, default = 200
        A parameter used for training epochs. 
    conv1_kernel_size: int, default = 13
        A parameter used for the kernel size of first covolutional layers
    dense_layers: list, default = [128]
        A parameter used for the dense layers.  
    batch_size: int, default: 128
        A parameter used for the batch size.
    lr: float, default: 1e-4
        A parameter used for the learning rate.
    batch_norm: bool, default: False
        batch normalization after convolution layers.
    n_inception: int, default:2
        Number of the inception layers.
    dense_avf: str, default is 'relu'
        activation fuction in the dense layers.
    dropout: float, default: 0
        A parameter used for the dropout of the dense layers, such as 0.1, 0.3, 0.5.
    monitor: str, default: 'val_loss'
        {'val_loss', 'val_auc'}, a monitor for model selection
    metric: str, default: 'ROC'
        {'ROC', 'ACC', 'PRC'},  a matric parameter
    patience: int, default = 10000, 
        A parameter used for early stopping
    gpuid: int, default=0,
        A parameter used for specific gpu card
    verbose: int, default = 0
    random_state, int, default: 32
    name: str 

    Examples
    --------
    >>> from aggmap import AggModel
    >>> clf = AggModel.MultiLabelEstimator()
    """
    
    def __init__(self, 
                 epochs = 200,  
                 conv1_kernel_size = 13,
                 dense_layers = [128],  
                 dense_avf = 'relu',
                 batch_size = 128,  
                 lr = 1e-4, 
                 batch_norm = False,
                 n_inception = 2,                     
                 dropout = 0.0,                 
                 monitor = 'val_loss', 
                 metric = 'ROC',
                 patience = 10000,
                 verbose = 0, 
                 random_state = 32,
                 name = "AggMap MultiLabels Estimator",
                 gpuid = 0,
                ):
        
        
        self.epochs = epochs
        self.dense_layers = dense_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.dense_avf = dense_avf
        self.batch_size = batch_size
        self.lr = lr
        self.batch_norm = batch_norm
        self.n_inception = n_inception
        self.dropout = dropout
        self.monitor = monitor
        self.metric = metric
        self.patience = patience
        self.gpuid = str(gpuid)
        os.environ["CUDA_VISIBLE_DEVICES"]= self.gpuid        
        
        self.verbose = verbose
        self.random_state = random_state
        self.is_fit = False        
        self.name = name
        
        print(self.get_params())
        
        
    def get_params(self, deep=True):

        model_paras =  {"epochs": self.epochs, 
                        "lr":self.lr, 
                        "conv1_kernel_size": self.conv1_kernel_size,
                        "dense_layers": self.dense_layers, 
                        "dense_avf":self.dense_avf, 
                        "batch_size":self.batch_size, 
                        "dropout":self.dropout,
                        "batch_norm":self.batch_norm,
                        "n_inception":self.n_inception,                        
                        "monitor": self.monitor,
                        "patience":self.patience,
                        "random_state":self.random_state,
                        "verbose":self.verbose,
                        "name":self.name,
                        "gpuid": self.gpuid,
                       }

        return model_paras
    
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

        

    def fit(self, X, y,  X_valid = None, y_valid = None):

        # Check that X and y have correct shape
        
        if  X.ndim != 4:
            raise ValueError("Found array X with dim %d. %s expected == 4." % (X.ndim, self.name))
            
        if  y.ndim != 2:
            raise ValueError("Found array y with dim %d. %s expected == 2." % (y.ndim, self.name))    
    
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        
        if (X_valid is None) | (y_valid is None):
            
            X_valid = X
            y_valid = y
        
        np.random.seed(self.random_state)
        tf.compat.v1.set_random_seed(self.random_state)

            
        model = aggmodel.net.AggMapNet2(X.shape[1:],
                                        n_outputs = y.shape[-1], 
                                        conv1_kernel_size = self.conv1_kernel_size,
                                        batch_norm = self.batch_norm,
                                        n_inception = self.n_inception,
                                        dense_layers = self.dense_layers, 
                                        dense_avf = self.dense_avf, 
                                        dropout = self.dropout,
                                        last_avf = None)
        
        opt = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
        model.compile(optimizer = opt, loss = aggmodel.loss.cross_entropy)
        performance = aggmodel.cbks.CLA_EarlyStoppingAndPerformance((X, y), 
                                                                    (X_valid, y_valid), 
                                                                    patience = self.patience, 
                                                                    criteria = self.monitor,
                                                                    metric = self.metric,  
                                                                    last_avf = None,
                                                                    verbose = self.verbose,)

        history = model.fit(X, y, 
                  batch_size=self.batch_size, 
                  epochs= self.epochs, verbose= 0, shuffle = True, 
                  validation_data = (X_valid, y_valid), 
                  callbacks=[performance]) 

        self._model = model
        self._performance = performance
        # Return the classifier
        self.history = history
        self.is_fit = True
        
        return self



    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        if  X.ndim != 4:
            raise ValueError("Found array X with dim %d. %s expected == 4." % (X.ndim, self.name))
        y_prob = self._performance.sigmoid(self._model.predict(X))
        return y_prob
    

    def predict(self, X):
        y_pred = np.round(self.predict_proba(X))
        return y_pred
    
    

    def score(self, X, y, scoring = 'accuracy', sample_weight=None):
        """Returns the score using the `scoring` option on the given
        test data and labels.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        scoring: str, please refer to: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """
        assert scoring in SCORERS.keys(), 'scoring is not in %s' % SCORERS.keys()
        scoring = get_scorer(scoring)

        return scoring(self, X, y, sample_weight=sample_weight)
    
    def plot_model(self, to_file='model.png', 
                   show_shapes=True, 
                   show_layer_names=True, 
                   rankdir='TB', 
                   expand_nested=False, 
                   dpi=96):
        if self.is_fit:
            tf.keras.utils.plot_model(self._model, 
                       to_file=to_file, 
                       show_shapes=show_shapes, 
                       show_layer_names=show_layer_names, 
                       rankdir=rankdir, 
                       expand_nested=expand_nested, 
                       dpi=dpi)
        else:
            print('Please fit first!')