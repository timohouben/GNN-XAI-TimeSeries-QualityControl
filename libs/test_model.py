import glob, os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, accuracy_score, roc_curve, auc
from libs.preprocessing_functions import create_batched_dataset
from libs.visualize import plot_roc_curves, extract_target_info

def select_threshold(predictions, anomaly_flags_true):
    # test different thresholds for the best MCC
    threshold_ranges = np.unique(np.round(predictions, 3))
    mcc_array = []
    for range1 in threshold_ranges:
        mcc_array.append(matthews_corrcoef(anomaly_flags_true, np.greater(predictions, range1)))
    threshold_max = threshold_ranges[np.argmax(mcc_array)]
    print('Max MCC: {:.3f} for threshold: {:.3f}'.format(np.max(mcc_array), threshold_max))
    return threshold_max

def calculate_threshold(model_config, preproc_config, val_dataset, wrapping_functions, model, baseline=False,
                        call_numb=tf.data.AUTOTUNE):
    calculate_threshold = model_config.calculate_threshold
    if preproc_config.ds_type == 'soilnet':
        anomaly_date_ind = int(model.model_info[0].numpy()/model.model_info[-1].numpy())
    else:
        anomaly_date_ind = model.model_info[0].numpy()
    
    if calculate_threshold:
        val_dataset_test = create_batched_dataset(val_dataset, preproc_config, shuffle=False, baseline=baseline)[0]
        predictions_val = model.predict(val_dataset_test.map(wrapping_functions[0], call_numb).prefetch(
            call_numb)).flatten()
        sensor_ids_val, anomaly_dates_val, anomaly_flags_true_val = extract_target_info(val_dataset_test.map(
                wrapping_functions[1], call_numb).prefetch(call_numb), anomaly_date_ind, ds_type=preproc_config.ds_type)
        threshold = select_threshold(predictions_val, anomaly_flags_true_val)
    else:
        threshold = 0.5
    return threshold, anomaly_date_ind, 
    
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
        plot_roc_curves([fpr], [tpr], model_config, [thr], [threshold], outpath, ['GCN'])
    else:
        if outpath is None:
            outpath = os.path.join(model_config.plotting.outdir, 'ROC_curve_baseline.png')
        plot_roc_curves([fpr], [tpr], model_config, [thr], [threshold], outpath, ['baseline'])
    return mcc, precision, recall, accuracy, auc_score, fpr, tpr, thr
