import numpy as np
import math
import tensorflow.keras.backend as K
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="functions")   
def matthews_correlation(y_true, y_pred):
    """
    Matthews correlation metric.
    It is only computed as a batch-wise average, not globally.
    Computes the Matthews correlation coefficient measure for quality
    of binary classifiers.
    """
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def tp(y_true, y_pred):
    y_pred_pos = K.round(y_pred)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(y_true)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)/K.sum(y_pos)
    return tp

def tn(y_true, y_pred):
    y_pred_pos = K.round(y_pred)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(y_true)
    y_neg = 1 - y_pos
    tn = K.sum(y_neg * y_pred_neg)/K.sum(y_neg)
    return tn

def fp(y_true, y_pred):
    y_pred_pos = K.round(y_pred)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(y_true)
    y_neg = 1 - y_pos
    fp = K.sum(y_neg * y_pred_pos)/K.sum(y_neg)
    return fp

def fn(y_true, y_pred):
    y_pred_pos = K.round(y_pred)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(y_true)
    y_neg = 1 - y_pos
    fn = K.sum(y_pos * y_pred_neg)/K.sum(y_pos)
    return fn


def Roc_curve(y_pred, y_true):
    roc = []

    y_unique = np.unique(np.round(y_pred, 7))
    
    for i in range(0, len(y_unique), 1):
        t = np.sort(y_unique)[i]         
        
        y_predicted=np.ravel(y_pred>t)  
        true_pos = np.sum(np.logical_and(y_true==1, y_predicted==1)).astype('float64')
        true_neg = np.sum(np.logical_and(y_true==0, y_predicted==0)).astype('float64')
        false_pos = np.sum(np.logical_and(y_true==0, y_predicted==1)).astype('float64')
        false_neg = np.sum(np.logical_and(y_true==1, y_predicted==0)).astype('float64')
        cond_neg = true_neg+false_pos
        cond_pos = true_pos+false_neg
        
        roc.append([true_pos/cond_pos,
                    false_pos/cond_neg])
    
    roc.append([0,0])

    return np.array(roc)

def PR_curve(y_pred, y_true):
    pr = []

    y_unique = np.unique(np.round(y_pred, 7))
    
    for i in range(0, len(y_unique), 1):
        t = np.sort(y_unique)[i]         
        
        y_predicted=np.ravel(y_pred>t)  
        true_pos = np.sum(np.logical_and(y_true==1, y_predicted==1)).astype('float64')
        true_neg = np.sum(np.logical_and(y_true==0, y_predicted==0)).astype('float64')
        false_pos = np.sum(np.logical_and(y_true==0, y_predicted==1)).astype('float64')
        false_neg = np.sum(np.logical_and(y_true==1, y_predicted==0)).astype('float64')
        cond_neg = true_neg+false_pos
        cond_pos = true_pos+false_neg
        
        pr.append([true_pos/(true_pos+false_pos),
                    true_pos/(true_pos+false_neg)])
    
#     pr.append([0,0])

    return np.array(pr)

def scorer(y_pred, y_true):
    pr = []
    roc = []
    mccs = []

    y_unique = np.unique(np.round(y_pred, 7))
    
    for i in range(0, len(y_unique), 1):
        t = np.sort(y_unique)[i]         
        
        y_predicted=np.ravel(y_pred>t)  
        true_pos = np.sum(np.logical_and(y_true==1, y_predicted==1)).astype('float64')
        true_neg = np.sum(np.logical_and(y_true==0, y_predicted==0)).astype('float64')
        false_pos = np.sum(np.logical_and(y_true==0, y_predicted==1)).astype('float64')
        false_neg = np.sum(np.logical_and(y_true==1, y_predicted==0)).astype('float64')
        cond_neg = true_neg+false_pos
        cond_pos = true_pos+false_neg
        if true_pos>0:
            pr.append([true_pos/(true_pos+false_pos),
                        true_pos/(true_pos+false_neg)])
        roc.append([true_pos/cond_pos,
                    false_pos/cond_neg])
        mccs.append([MCC(y_true, y_predicted),t])
    
    roc.append([0,0])
    
#     pr.append([0,0])

    return np.array(pr), np.array(roc), np.array(mccs), 


def MCC(y_true, y_pred):
    """
    Computes the Matthews correlation coefficient measure for quality
    of binary classification problems.
    """

    tp = np.sum(np.logical_and(y_true==1, y_pred==1))
    tn = np.sum(np.logical_and(y_true==0, y_pred==0))
    fp = np.sum(np.logical_and(y_true==0, y_pred==1))
    fn = np.sum(np.logical_and(y_true==1, y_pred==0))

    numerator = (tp * tn - fp * fn)
    denominator = math.sqrt(tp + fp) * math.sqrt(tp + fn) * math.sqrt(tn + fp) * math.sqrt(tn + fn)

    return numerator / (denominator + 1e-07)

def AUC(roc):
    k = len(roc)
    auc=0
    for i in range(k-1):
        auc= auc+(np.abs(roc[i,1]-roc[i+1,1]))*0.5*(roc[i+1,0]+roc[i,0])
    
    return auc


def mse(truth, predict, config, norm=None, func=None):
    
    if func == True:
        
        x = np.arange(0,1,1/config.window)
        x = x.reshape(config.window,1)
        y = (truth - predict)*x
    else:
        y = (truth - predict)
    
    
    if truth.shape[2] < 2:
    
        mse = np.concatenate(np.mean(np.power(y, 2), axis=1))
        
    else:
        
        mse = np.mean(np.mean(np.power(y, 2), axis=1), axis=1)
        

    
    if norm == True:
        
        mse = (mse - np.min(mse)) / (np.max(mse) - np.min(mse))
    
    return mse

@tf.keras.utils.register_keras_serializable(package="functions") 
def mcc_metric(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, -1), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, -1), tf.float32)
    threshold = 0.5  
    predicted = tf.cast(tf.greater(y_pred, threshold), tf.float32)
    true_pos = tf.math.count_nonzero(predicted * y_true)
    true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
    false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
    false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
    #print(true_pos, true_neg, false_pos, false_neg)
    #denominator = 
    #numerator = 
    #print('numerator', numerator)
    #print('denominator', denominator)
    mcc = tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float64) / (tf.sqrt(tf.cast((true_pos + false_pos) * (true_pos + false_neg) 
      * (true_neg + false_pos) * (true_neg + false_neg), tf.float64)) + 1e-07)
    return mcc

