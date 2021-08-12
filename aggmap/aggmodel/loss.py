import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np


###########  classification ##############
def cross_entropy(y_true, y_pred):
    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return cost


def weighted_cross_entropy(y_true, y_pred, pos_weight):
  
    cost = tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=logits, 
                                                    pos_weight = pos_weight)
    return cost





def MALE(y_obs, y_pred):
    return tf.keras.backend.log(0.5 + tf.keras.backend.abs(y_pred - y_obs))