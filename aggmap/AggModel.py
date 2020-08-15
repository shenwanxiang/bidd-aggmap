import warnings
warnings.filterwarnings("ignore")


import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import get_scorer, SCORERS


from aggmap import aggmodel




class RegressionEstimator(BaseEstimator, RegressorMixin):
    """ An AggMap CNN MultiClass estimator (each sample belongs to only one class) 
    Parameters
    ----------
    epochs : int, default = 100
        A parameter used for training epochs. 
    dense_layers: list, default = [128]
        A parameter used for the dense layers.    
    monitor: str
        {'val_loss', 'val_r2'}
        
    
    Examples
    --------
    >>> from aggmap import AggModel
    >>> clf = AggModel.RegressionEstimator()

    """
    
    def __init__(self, 
                 epochs = 100,  
                 conv1_kernel_size = 3,
                 dense_layers = [128],  
                 dense_avf = 'relu',
                 batch_size = 128,  
                 lr = 1e-4, 
                 monitor = 'val_loss', 
                 metric = 'r2',
                 patience = 10000,
                 verbose = 0, 
                 random_state = 32,
                 name = "AggMap Regression Estimator"
                ):
        
        
        self.epochs = epochs
        self.dense_layers = dense_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.dense_avf = dense_avf
        self.batch_size = batch_size
        self.lr = lr
        self.monitor = monitor
        self.metric = metric
        self.patience = patience
        
        
        self.verbose = verbose
        self.random_state = random_state
        
        self.name = name
        
        print(self)
        
        
    def get_params(self, deep=True):

        model_paras =  {"epochs": self.epochs, 
                        "lr":self.lr, 
                        "conv1_kernel_size": self.conv1_kernel_size,
                        "dense_layers": self.dense_layers, 
                        "dense_avf":self.dense_avf, 
                        "batch_size":self.batch_size, 
                        "monitor": self.monitor,
                        "patience":self.patience,
                        "random_state":self.random_state,
                        "verbose":self.verbose,
                        "name":self.name,
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
        
        if (X_valid == None) | (y_valid == None):
            
            X_valid = X
            y_valid = y
        
        np.random.seed(self.random_state)
        tf.compat.v1.set_random_seed(self.random_state)
        
        model = aggmodel.net.AggMapNet(X.shape[1:],
                                       n_outputs = y.shape[-1], 
                                       conv1_kernel_size = self.conv1_kernel_size,
                                       dense_layers = self.dense_layers, 
                                       dense_avf = self.dense_avf, 
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
    
    
    
    
    
class MultiClassesEstimator(BaseEstimator, ClassifierMixin):

    """ An AggMap CNN MultiClass estimator (each sample belongs to only one class) 
    Parameters
    ----------
    epochs : int, default = 150
        A parameter used for training epochs. 
    dense_layers: list, default = [128]
        A parameter used for the dense layers.    
    
    Examples
    --------
    >>> from aggmap import aggmodel

    """
    
    def __init__(self, 
                 epochs = 150,  
                 conv1_kernel_size = 3,
                 dense_layers = [128],  
                 dense_avf = 'relu',
                 batch_size = 128,  
                 lr = 1e-4, 
                 monitor = 'val_loss', 
                 metric = 'ROC',
                 patience = 10000,
                 verbose = 0, 
                 random_state = 32,
                 name = "AggMap MultiClass Estimator"
                ):
        
        
        self.epochs = epochs
        self.dense_layers = dense_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.dense_avf = dense_avf
        self.batch_size = batch_size
        self.lr = lr
        self.monitor = monitor
        self.metric = metric
        self.patience = patience
        
        
        self.verbose = verbose
        self.random_state = random_state
        
        self.name = name
        
        print(self)
        
        
    def get_params(self, deep=True):

        model_paras =  {"epochs": self.epochs, 
                        "lr":self.lr, 
                        "conv1_kernel_size": self.conv1_kernel_size,
                        "dense_layers": self.dense_layers, 
                        "dense_avf":self.dense_avf, 
                        "batch_size":self.batch_size, 
                        "monitor": self.monitor,
                        "metric":self.metric,
                        "patience":self.patience,
                        "random_state":self.random_state,
                        "verbose":self.verbose,
                        "name":self.name,
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
        
        if (X_valid == None) | (y_valid == None):
            
            X_valid = X
            y_valid = y
        
        np.random.seed(self.random_state)
        tf.compat.v1.set_random_seed(self.random_state)
        
        model = aggmodel.net.AggMapNet(X.shape[1:],
                                       n_outputs = y.shape[-1], 
                                       conv1_kernel_size = self.conv1_kernel_size,
                                       dense_layers = self.dense_layers, 
                                       dense_avf = self.dense_avf, 
                                       last_avf = 'softmax')

        
        opt = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
        model.compile(optimizer = opt, loss = aggmodel.loss.cross_entropy)
        performance = aggmodel.cbks.CLA_EarlyStoppingAndPerformance((X, y), 
                                                                    (X_valid, y_valid), 
                                                                    patience = self.patience, 
                                                                    criteria = self.monitor,
                                                                    metric = self.metric,  
                                                                    last_avf="softmax",
                                                                    verbose = self.verbose,)

        model.fit(X, y, 
                  batch_size=self.batch_size, 
                  epochs= self.epochs, verbose= 0, shuffle = True, 
                  validation_data = (X_valid, y_valid), 
                  callbacks=[performance]) 

        self._model = model
        self._performance = performance
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
    
    
    
class MultiLabelsEstimator(BaseEstimator, ClassifierMixin):

    """ An AggMap CNN MultiLabel estimator (each sample belongs to many classes) 
    Parameters
    ----------
    epochs : int, default = 150
        A parameter used for training epochs. 
    dense_layers: list, default = [128]
        A parameter used for the dense layers.    
    
    Examples
    --------
    >>> from aggmap import aggmodel

    """
    
    def __init__(self, 
                 epochs = 150,  
                 conv1_kernel_size = 3,
                 dense_layers = [128],  
                 dense_avf = 'relu',
                 batch_size = 128,  
                 lr = 1e-4, 
                 monitor = 'val_loss', 
                 metric = 'ROC',
                 patience = 10000,
                 verbose = 0, 
                 random_state = 32,
                 name = "AggMap MultiLabels Estimator"
                ):
        
        
        self.epochs = epochs
        self.dense_layers = dense_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.dense_avf = dense_avf
        self.batch_size = batch_size
        self.lr = lr
        self.monitor = monitor
        self.metric = metric
        self.patience = patience
        
        
        self.verbose = verbose
        self.random_state = random_state
        
        self.name = name
        
        print(self)
        
        
    def get_params(self, deep=True):

        model_paras =  {"epochs": self.epochs, 
                        "lr":self.lr, 
                        "conv1_kernel_size": self.conv1_kernel_size,
                        "dense_layers": self.dense_layers, 
                        "dense_avf":self.dense_avf, 
                        "batch_size":self.batch_size, 
                        "monitor": self.monitor,
                        "metric":self.metric,
                        "patience":self.patience,
                        "random_state":self.random_state,
                        "verbose":self.verbose,
                        "name":self.name,
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
        
        if (X_valid == None) | (y_valid == None):
            
            X_valid = X
            y_valid = y
        
        np.random.seed(self.random_state)
        tf.compat.v1.set_random_seed(self.random_state)
        
        model = aggmodel.net.AggMapNet(X.shape[1:],
                                       n_outputs = y.shape[-1], 
                                       conv1_kernel_size = self.conv1_kernel_size,
                                       dense_layers = self.dense_layers, 
                                       dense_avf = self.dense_avf, 
                                       last_avf = None)

        
        opt = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
        model.compile(optimizer = opt, loss = aggmodel.loss.cross_entropy)
        performance = aggmodel.cbks.CLA_EarlyStoppingAndPerformance((X, y), 
                                                                    (X_valid, y_valid), 
                                                                    patience = self.patience, 
                                                                    criteria = self.monitor,
                                                                    metric = self.metric,  
                                                                    last_avf=None,
                                                                    verbose = self.verbose,)

        model.fit(X, y, 
                  batch_size=self.batch_size, 
                  epochs= self.epochs, verbose= 0, shuffle = True, 
                  validation_data = (X_valid, y_valid), 
                  callbacks=[performance]) 

        self._model = model
        self._performance = performance
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
