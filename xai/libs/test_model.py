import glob, os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, accuracy_score, roc_curve, auc
from libs.visualize import plot_roc_curve


def select_threshold(predictions, anomaly_flags_true):
    # test different thresholds for the best MCC
    threshold_ranges = np.unique(np.round(predictions, 3))
    mcc_array = []
    for range1 in threshold_ranges:
        mcc_array.append(matthews_corrcoef(anomaly_flags_true, np.greater(predictions, range1)))
    threshold_max = threshold_ranges[np.argmax(mcc_array)]
    print('Max MCC: {:.3f} for threshold: {:.3f}'.format(np.max(mcc_array), threshold_max))
    return threshold_max


def calculate_metrics(anomaly_flags_true, anomaly_flags_pred, predictions, model_config, threshold=0.5, baseline=False, outpath=None):
    # MCC
    mcc = matthews_corrcoef(anomaly_flags_true, anomaly_flags_pred)
    # Precision
    precision = precision_score(anomaly_flags_true, anomaly_flags_pred)
    # Recall
    recall = recall_score(anomaly_flags_true, anomaly_flags_pred)
    # Accuracy
    accuracy = accuracy_score(anomaly_flags_true, anomaly_flags_pred)
    fpr, tpr, thr = roc_curve(anomaly_flags_true, predictions)
    auc_score = auc(fpr, tpr)
    print("MCC: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}\nAccuracy: {:.3f}\nAUC: {:.3f} ".format(
            mcc, precision, recall, accuracy, auc_score))
    if baseline == False:
        if outpath is None:
            outpath = os.path.join(model_config.plotting.outdir, 'ROC_curve.png')
        plot_roc_curve([fpr], [tpr], model_config, [thr], [threshold], outpath, ['GCN'])
    else:
        if outpath is None:
            outpath = os.path.join(model_config.plotting.outdir, 'ROC_curve_baseline.png')
        plot_roc_curve([fpr], [tpr], model_config, [thr], [threshold], outpath, ['baseline'])
    return mcc, precision, recall, accuracy, auc_score, fpr, tpr, thr
