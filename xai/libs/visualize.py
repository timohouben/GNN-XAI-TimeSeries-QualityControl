import os, glob
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import xarray as xr
from libs.preprocessing_functions import wrapper_batch_mapping_plot, wrapper_batch_mapping, wrapper_batch_mapping_baseline, wrapper_batch_mapping_baseline_plot
from matplotlib.lines import Line2D
from sklearn.metrics import auc
from scipy import interpolate
import matplotlib.dates as mdates


def extract_target_info(dataset, anomaly_date_ind):
    anomaly_dates = []
    anomaly_flags_true = []
    sensor_ids = []
    for x in dataset:
        anomaly_dates.append(pd.to_datetime([curr_date.decode('utf-8') for curr_date in x[0][-1].numpy()[:, anomaly_date_ind]],
                                            format='%Y-%m-%dT%H:%M:%S'))
        try:
            sensor_ids.append([sensor_id.decode('utf-8') for sensor_id in x[0][-2].numpy()])
        except AttributeError: 
            sensor_ids.append([sensor_id for sensor_id in x[0][-2].numpy()])

        anomaly_flags_true.append(x[1].numpy())
    anomaly_flags_true = np.concatenate(anomaly_flags_true)
    sensor_ids = np.concatenate(sensor_ids)
    anomaly_dates = np.concatenate(anomaly_dates)
    return sensor_ids, anomaly_dates, anomaly_flags_true


def classified_timeseries_figure(sensor_id, features_timeseries, feature_timeseries_dates, anomaly_flags_true,
                                 anomaly_flags_pred, model_config, showfig=False):
    # anomaly_flags_dates,
    alpha = model_config.plotting.alpha
    # Plot figures
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_title('{}'.format(cml_id), pad=10)
    ymin = np.nanmin(features_timeseries) - 1
    ymax = np.nanmax(features_timeseries) + 5
    ax.set_ylim([ymin, ymax])
    ax.fill_between(feature_timeseries_dates, ymin, ymax,
                            where=np.logical_and(anomaly_flags_pred==1, anomaly_flags_true==1),
                            label='True Positive', alpha=alpha, color='green')
    ax.fill_between(feature_timeseries_dates, ymin, ymax,
                            where=np.logical_and(anomaly_flags_pred==0, anomaly_flags_true==0),
                            label='True Negative', alpha=alpha, color='blue')
    ax.fill_between(feature_timeseries_dates, ymin, ymax,
                            where=np.logical_and(anomaly_flags_pred==0, anomaly_flags_true==1),
                            label='False Negative', alpha=alpha, color='red')
    ax.fill_between(feature_timeseries_dates, ymin, ymax,
                            where=np.logical_and(anomaly_flags_pred==1, anomaly_flags_true==0),
                            label='False Positive', alpha=alpha, color='orange')
    for feature_timeseries in features_timeseries:
        ax.plot(feature_timeseries_dates, feature_timeseries)
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend()
    outpath = os.path.join(model_config.plotting.outdir, '{}_start_{}_end_{}.png'.format(sensor_id, np.datetime_as_string(feature_timeseries_dates[0], unit='m'),                                                                            np.datetime_as_string(feature_timeseries_dates[-1], unit='m')))
    plt.margins(x=0.001)
    plt.savefig(outpath, bbox_inches='tight')
    if showfig:
        plt.show()
    else:
        plt.close()
    

def plot_classified_timeseries(model, test_dataset, mapping_functions, raw_dataset, model_config, predictions=None, start_date=None, end_date=None, baseline=False, ds_type='cml', showfig=True, threshold = 0.5):

    if not os.path.exists(model_config.plotting.outdir):
        os.makedirs(model_config.plotting.outdir)
    call_numb = tf.data.AUTOTUNE
    if predictions is None:
        if baseline==False:
            predictions = model.predict(test_dataset.map(mapping_functions[0], call_numb).prefetch(call_numb)).flatten()
        else:
            predictions = model.predict(test_dataset.map(mapping_functions[0], call_numb).prefetch(call_numb)).flatten()
    anomaly_flags_pred = np.where(predictions > threshold, 1, 0)
    # plot figures with h interval
    interval = model_config.plotting.plot_time_range
    if start_date == None:
        start_date = raw_dataset.time.min().values
    if end_date == None:
        end_date = raw_dataset.time.max().values 
    time_ranges = pd.date_range(start=start_date, end=end_date + np.timedelta64(interval, 'h'), freq='%dH' % interval)
    if ds_type == 'cml':
        anomaly_date_ind = model.model_info[0].numpy()
    elif ds_type == 'soilnet':
        anomaly_date_ind = int(model.model_info[0].numpy()/model.model_info[-1].numpy())
        
    if baseline==False:
        sensor_ids, anomaly_dates, anomaly_flags_true = extract_target_info(test_dataset.map(
            mapping_functions[1], call_numb).prefetch(call_numb), anomaly_date_ind)
    else:
        sensor_ids, anomaly_dates, anomaly_flags_true = extract_target_info(test_dataset.map(mapping_functions[1], call_numb).prefetch(call_numb), anomaly_date_ind)
    
    for sensor_id in np.unique(sensor_ids):
        print(sensor_id)
        ds_sensor = raw_dataset.sel(sensor_id=sensor_id)
        # select observations for the current sensor 
        ind_sensor = np.where(sensor_ids==sensor_id)
        for i in range(len(time_ranges)-1):
            ds_sensor_date_range = ds_sensor.sel(time=slice(time_ranges[i], time_ranges[i+1]))
            plot_dates = ds_sensor_date_range.time.values
            # select true and predicted labels
            ind_dates = np.where((anomaly_dates>=time_ranges[i]) & (anomaly_dates<=time_ranges[i+1]))
            curr_ind = np.intersect1d(ind_sensor, ind_dates)
            if len(curr_ind)==0:
                continue
            curr_pred_anomaly = anomaly_flags_pred[curr_ind]
            curr_true_anomaly = anomaly_flags_true[curr_ind]
            curr_anomaly_dates = anomaly_dates[curr_ind]

            pred_anomalies_timeseries = np.empty(len(plot_dates))
            pred_anomalies_timeseries[:] = np.nan
            true_anomalies_timeseries = np.copy(pred_anomalies_timeseries)
            _, plot_ind, anomaly_ind = np.intersect1d(plot_dates, curr_anomaly_dates, return_indices=True)
            pred_anomalies_timeseries[plot_ind] = curr_pred_anomaly[anomaly_ind]
            true_anomalies_timeseries[plot_ind] = curr_true_anomaly[anomaly_ind]
            if ds_type == 'cml':
                trsl1 = ds_sensor_date_range.TL_1.values
                trsl2 = ds_sensor_date_range.TL_2.values
                features = [trsl1, trsl2]
            elif ds_type == 'soilnet':
                features = [ds_sensor_date_range.moisture.values]
            classified_timeseries_figure(sensor_id, features, plot_dates, true_anomalies_timeseries,
                                         pred_anomalies_timeseries, model_config, showfig=showfig)


def plot_roc_curve(fpr, tpr, model_config, thresholds=None, choosen_thresholds=None,
                   outpath=None, labels=['GCN', 'baseline']):
    color_tab = ['cornflowerblue', 'sandybrown', 'green']
    if outpath is None:
        outpath = os.path.join(model_config.plotting.outdir, 'ROC_curve.png')
    
    for i in range(len(fpr)):
        auc_score = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = {:.3f}) {:s}'.format(auc_score, labels[i]),
                color=color_tab[i])
        if thresholds is not None:
            tpr_intrp = interpolate.interp1d(thresholds[i], tpr[i])
            fpr_intrp = interpolate.interp1d(thresholds[i], fpr[i])
            tpr_thr = tpr_intrp(choosen_thresholds[i])
            fpr_thr = fpr_intrp(choosen_thresholds[i])
            plt.plot(fpr_thr, tpr_thr, 'o', color=color_tab[i])
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.margins(x=0.001)
    plt.savefig(outpath, bbox_inches='tight')
    plt.show()



def timeseries_figure(predicted, true, cml_timeseries, sensor_id, dates, outdir, anomaly_time_ind,
                      model_config, showfig=False, predictions=None):
    alpha = model_config.plotting.alpha
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    curr_date = dates[anomaly_time_ind]
    # select minimum value
    ymin = np.nanmin(cml_timeseries) - 1
    ymax = np.nanmax(cml_timeseries) + 1
    ax.set_ylim([ymin, ymax])
    ax.set_title('{} on {}'.format(sensor_id, curr_date), pad=10)
    if (predicted == 0) and (true == 0):
        color = 'blue'
    elif (predicted == 1) and (true == 0):
        color = 'orange'
    elif (predicted == 1) and (true == 1):
        color='green'
    elif (predicted == 0) and (true == 1):
        color='red'
    ax.fill_between(dates[anomaly_time_ind-1:anomaly_time_ind+1], ymin, ymax,
                            alpha=alpha, color=color)
    legend_elements = [Line2D([0], [0], color='green', ls='-',lw=2, label='True Positive'),
                       Line2D([0], [0], color='blue', ls='-',lw=2, label='True Negative'),
                   Line2D([0], [0], color='orange', ls='-',lw=2, label='False Positive'),        
                   Line2D([0], [0], color='red', ls='-',lw=2, label='False Negative')] 
    ax.plot(dates, cml_timeseries[:, 0])
    ax.plot(dates, cml_timeseries[:, 1])
    ax.legend(handles=legend_elements)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    if predictions is not None:
        ax2 = ax.twinx()
        ax2.set_ylim([0, 1])
        ax2.plot(dates, predictions, 'k', label='probability', linewidth=0.5)
        ax2.set_ylabel('probability')
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=0, horizontalalignment='right') #, fontsize=40
    # check if output dir is path to a file, if not then create filename
    _, file_extension = os.path.splitext(outdir)
    if file_extension:
        outpath = outdir
    else:
        outpath = os.path.join(outdir, '{}_{}_true_{}_pred_{}'.format(cml_id, curr_date, true, predicted))
    plt.margins(x=0.001)
    plt.savefig(outpath, bbox_inches='tight')
    if showfig:
        plt.show()
    else:
        plt.close()
    return ax

def plot_classified_samples(model, tfrecords_dataset, model_config, threshold=0.5, baseline=False, out_dir=None):
    anomaly_time_ind = model.model_info[0].numpy()
    if out_dir is None:
        if baseline == True:
            out_dir = os.path.join(model_config.plotting.outdir, 'classified_validation_samples_baseline')
        else:
            out_dir = os.path.join(model_config.plotting.outdir, 'classified_validation_samples')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for features, anomaly_flags_true in tfrecords_dataset:
        sensor_dates = features[-1]
        sensor_ids = features[-2]
        # predict anomalies
        if baseline == True:
            sensor_anomaly_timeseries = features[0]
            anomaly_flags_pred = model(features[0], training=False)
        else:
            sensor_anomaly_timeseries = features[1]
            #anomaly_flags_pred = model(features[:9], training=False)
            anomaly_flags_pred = model(features[:-3], training=False)
        anomaly_flags_pred_class = np.where(anomaly_flags_pred > threshold, 1, 0)
        for b in range(sensor_dates.shape[0]):
            curr_dates = pd.to_datetime([x.decode('utf-8') for x in tf.squeeze(sensor_dates[b]).numpy()], format='%Y-%m-%dT%H:%M:%S')
            curr_sensor_id = tf.squeeze(sensor_ids[b]).numpy().decode('utf-8')
            timeseries_figure(anomaly_flags_pred_class[b][0], anomaly_flags_true[b],
                              sensor_anomaly_timeseries[b], curr_sensor_id, curr_dates, out_dir, anomaly_time_ind, model_config)


def classified_timeseries_figure_with_neighbours(sensor_ids, all_features_timeseries,
                                                 feature_timeseries_dates,
                                                 anomaly_flags_true, anomaly_flags_pred, model_config,
                                                 flags, probabilities=None, distances=None, ymin=None,
                                                 ymax=None, showfig=False):
    alpha = model_config.plotting.alpha
    sensor_numb = len(sensor_ids)
    # Plot figures
    fig, ax = plt.subplots(sensor_numb, 1, figsize=(20, sensor_numb*3.5), sharex='all')
    if ymin is None:
        calculate = True
    
    for i in range(sensor_numb):
        features_timeseries = all_features_timeseries[i, :, :]
        sensor_id = sensor_ids[i]
        if calculate is True:
            ymin = np.nanmin(features_timeseries) - 1
            #if ymax is None:
            ymax = ymin + 25
            #ymax = np.nanmax(features_timeseries) + 5
        ax[i].set_ylim([ymin, ymax])
        ax[i].fill_between(feature_timeseries_dates, ymin, ymax,
                                where=np.logical_and(anomaly_flags_pred==1, anomaly_flags_true==1),
                                label='True Positive', alpha=alpha, color='green')
        ax[i].fill_between(feature_timeseries_dates, ymin, ymax,
                                where=np.logical_and(anomaly_flags_pred==0, anomaly_flags_true==0),
                                label='True Negative', alpha=alpha, color='blue')
        ax[i].fill_between(feature_timeseries_dates, ymin, ymax,
                                where=np.logical_and(anomaly_flags_pred==0, anomaly_flags_true==1),
                                label='False Negative', alpha=alpha, color='red')
        ax[i].fill_between(feature_timeseries_dates, ymin, ymax,
                                where=np.logical_and(anomaly_flags_pred==1, anomaly_flags_true==0),
                                label='False Positive', alpha=alpha, color='orange')
        if flags[i]:
            ax[i].set_title('Anomalous sensor: {}'.format(sensor_id), pad=10, fontweight='bold')
            flagged_sensor_id = sensor_id
        else:
            if distances is not None:
                distance = distances[i]
                ax[i].set_title('Neighbouring sensor: {} distance: {:.1f}'.format(sensor_id, distance), pad=10)
            else:
                ax[i].set_title('Neighbouring sensor: {}'.format(sensor_id), pad=10)
        if i == sensor_numb-1:
            ax[i].legend()
        
        if probabilities is not None:
            ax2 = ax[i].twinx()
            ax2.set_ylim([0, 1])
            ax2.plot(feature_timeseries_dates, probabilities, 'k', label='probability', linewidth=0.5)
        for t in range(features_timeseries.shape[-1]):
            ax[i].plot(feature_timeseries_dates, features_timeseries[:, t])
       
        ax[i].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax[i].xaxis.set_major_locator(mdates.HourLocator(interval=24))
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
    out_dir = os.path.join(model_config.plotting.outdir, 'classification_with_neighbors')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outpath = os.path.join(out_dir, '{}_start_{}_end_{}_neighbours.png'.format(flagged_sensor_id,
                        np.datetime_as_string(feature_timeseries_dates[0], unit='m'),                                                                            np.datetime_as_string(feature_timeseries_dates[-1], unit='m')))
    plt.margins(x=0.001)
    plt.savefig(outpath, bbox_inches='tight')
    if showfig:
        plt.show()
    else:
        plt.close()


def classified_timeseries_figure_comparison(sensor_id, features_timeseries, feature_timeseries_dates, anomaly_flags_true,
                                 anomaly_flags_pred, model_config, out_dir='../plots', probabilities=None,
                                            labels=['GCN', 'Baseline'], showfig=False):
    alpha = model_config.plotting.alpha
    # Plot figures
    fig, ax = plt.subplots(len(anomaly_flags_true), 1, figsize=(20, 3.5*len(anomaly_flags_true)), sharex='all')
    fig.suptitle('{}'.format(sensor_id), y=0.99, fontweight='bold')
    for i in range(len(anomaly_flags_pred)):
        if len(anomaly_flags_pred) == 1:
            curr_ax = ax
        else:   
            curr_ax = ax[i]
        ymin = np.nanmin(features_timeseries) - 1
        ymax = np.nanmax(features_timeseries) + 5
        curr_ax.set_ylim([ymin, ymax])
        curr_ax.fill_between(feature_timeseries_dates, ymin, ymax,
                                where=np.logical_and(anomaly_flags_pred[i]==1, anomaly_flags_true[i]==1),
                                label='True Positive', alpha=alpha, color='green')
        curr_ax.fill_between(feature_timeseries_dates, ymin, ymax,
                                where=np.logical_and(anomaly_flags_pred[i]==0, anomaly_flags_true[i]==0),
                                label='True Negative', alpha=alpha, color='blue')
        curr_ax.fill_between(feature_timeseries_dates, ymin, ymax,
                                where=np.logical_and(anomaly_flags_pred[i]==0, anomaly_flags_true[i]==1),
                                label='False Negative', alpha=alpha, color='red')
        curr_ax.fill_between(feature_timeseries_dates, ymin, ymax,
                                where=np.logical_and(anomaly_flags_pred[i]==1, anomaly_flags_true[i]==0),
                                label='False Positive', alpha=alpha, color='orange')
        if labels[i]:
            curr_ax.set_title(labels[i], pad=5, fontweight='bold')
        curr_ax.legend()
        if probabilities is not None:
            ax2 = curr_ax.twinx()
            ax2.set_ylim([0, 1])
            ax2.plot(feature_timeseries_dates, probabilities[i], 'k', label='probability', linewidth=0.5)
        for feature_timeseries in features_timeseries:
            curr_ax.plot(feature_timeseries_dates, feature_timeseries)
       
        curr_ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        curr_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        curr_ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
    outpath = os.path.join(out_dir, '{}_start_{}_end_{}.png'.format(sensor_id,
                            np.datetime_as_string(feature_timeseries_dates[0], unit='m'),                                                                            np.datetime_as_string(feature_timeseries_dates[-1], unit='m')))
    plt.margins(x=0.001)
    plt.savefig(outpath, bbox_inches='tight')
    if showfig:
        plt.show()
    else:
        plt.close()


def plot_results(sensor_ids, anomaly_dates, anomaly_flags_pred, anomaly_flags_true, predictions,
                 preproc_config, model_config, plot_neighbors=False, comparison=False,
                 sensor_ids_baseline=None, anomaly_dates_baseline=None,
                 anomaly_flags_pred_baseline=None, anomaly_flags_true_baseline=None, predictions_baseline=None,
                labels=['GCN', 'baseline']):
    # plot figures with h interval
    interval = model_config.plotting.plot_time_range
    if comparison:
        out_dir = os.path.join(model_config.plotting.outdir, 'classified_timeseries_comparison')
    else:
        out_dir = os.path.join(model_config.plotting.outdir, 'classified_timeseries')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for sensor_id in np.unique(sensor_ids):
        # Open file for the sensor
        if preproc_config.ds_type == 'cml':
            data_path = os.path.join(preproc_config.dataset.ncfiles_dir, '{}.nc'.format(sensor_id))
        elif preproc_config.ds_type == 'soilnet':
            data_path = glob.glob(os.path.join(preproc_config.dataset.ncfiles_dir, '*_{}.nc'.format(sensor_id)))[0]
            
        ds_sensor = xr.open_dataset(data_path)
        start_date = ds_sensor.time.min().values
        end_date = ds_sensor.time.max().values 
        time_ranges = pd.date_range(start=start_date, end=end_date + np.timedelta64(interval, 'h'), freq='%dH' % interval)
        # select observations for the current sensor 
        ind_sensor = np.where(sensor_ids==sensor_id)
        if comparison:
            ind_sensor_baseline = np.where(sensor_ids_baseline==sensor_id)
        for i in np.arange(len(time_ranges)-1):
            ds_sensor_date_range = ds_sensor.sel(time=slice(time_ranges[i], time_ranges[i+1]))
            if preproc_config.ds_type == 'cml':
                missing_data_sensor_id = np.where(np.isnan(ds_sensor_date_range.TL_1).all(axis=1) &
                                       np.isnan(ds_sensor_date_range.TL_2).all(axis=1))[0]
            else:
                missing_data_sensor_id = np.where(np.isnan(ds_sensor_date_range.moisture).all(axis=1) &
                                       np.isnan(ds_sensor_date_range.battv).all(axis=1) &
                                                 np.isnan(ds_sensor_date_range.temp).all(axis=1))[0]
                
            if len(missing_data_sensor_id) > 0:
                ds_sensor_date_range = ds_sensor_date_range.drop_isel(sensor_id=missing_data_sensor_id)
            plot_dates = ds_sensor_date_range.time.values
            # select true and predicted labels
            ind_dates = np.where((anomaly_dates>=time_ranges[i]) & (anomaly_dates<=time_ranges[i+1]))
            curr_ind = np.intersect1d(ind_sensor, ind_dates)
            if len(curr_ind)==0:
                continue
            curr_pred_anomaly = anomaly_flags_pred[curr_ind]
            curr_true_anomaly = anomaly_flags_true[curr_ind]
            curr_probabilities = predictions[curr_ind]
            curr_anomaly_dates = anomaly_dates[curr_ind]
            
            pred_anomalies_timeseries = np.empty(len(plot_dates))
            pred_anomalies_timeseries[:] = np.nan
            true_anomalies_timeseries = np.copy(pred_anomalies_timeseries)
            pred_probabilities_timeseries = np.copy(pred_anomalies_timeseries)
            
            _, plot_ind, anomaly_ind = np.intersect1d(plot_dates, curr_anomaly_dates, return_indices=True)
            pred_anomalies_timeseries[plot_ind] = curr_pred_anomaly[anomaly_ind]
            true_anomalies_timeseries[plot_ind] = curr_true_anomaly[anomaly_ind]
            pred_probabilities_timeseries[plot_ind] = curr_probabilities[anomaly_ind]
            if preproc_config.ds_type == 'cml':
                trsl1 = ds_sensor_date_range.where(ds_sensor_date_range.flagged, drop=True).TL_1.values.flatten()
                trsl2 = ds_sensor_date_range.where(ds_sensor_date_range.flagged, drop=True).TL_2.values.flatten() 
                sample_trsl1 = ds_sensor_date_range.TL_1.values
                sample_trsl2 = ds_sensor_date_range.TL_2.values
                all_features_timeseries = np.stack([sample_trsl1, sample_trsl2], axis=-1)
                sensor_feature_timeseries = [trsl1, trsl2]
            elif preproc_config.ds_type == 'soilnet':
                sensor_feature_timeseries = [ds_sensor_date_range.where(
                    ds_sensor_date_range.flagged, drop=True).moisture.values.flatten()]
                all_features_timeseries = y = np.expand_dims(ds_sensor_date_range.moisture.values, -1)
            sample_sensor_ids = ds_sensor_date_range.sensor_id.values
            min_feature = None
            max_feature = None
            flags = ds_sensor_date_range.flagged.values
            distances = ds_sensor_date_range.distances.values[0, :]
            if plot_neighbors:
                classified_timeseries_figure_with_neighbours(sample_sensor_ids, all_features_timeseries, plot_dates,
                                                 true_anomalies_timeseries, pred_anomalies_timeseries, model_config, flags, 
                                                 probabilities=pred_probabilities_timeseries, distances=distances,
                                                    ymin=min_feature, ymax=max_feature, showfig=True)
            true_anomalies_array = [true_anomalies_timeseries]
            pred_anomalies_array = [pred_anomalies_timeseries]
            probabilities_array = [pred_probabilities_timeseries]
            
            if comparison:
                ind_dates_baseline = np.where((anomaly_dates_baseline>=time_ranges[i]) &
                                          (anomaly_dates_baseline<=time_ranges[i+1]))
                curr_ind_baseline = np.intersect1d(ind_sensor_baseline, ind_dates_baseline)
                curr_pred_anomaly_baseline = anomaly_flags_pred_baseline[curr_ind_baseline]
                curr_true_anomaly_baseline = anomaly_flags_true_baseline[curr_ind_baseline]
                curr_probabilities_baseline = predictions_baseline[curr_ind_baseline]
                curr_anomaly_dates_baseline = anomaly_dates_baseline[curr_ind_baseline]
                pred_anomalies_timeseries_baseline = np.copy(pred_anomalies_timeseries)
                true_anomalies_timeseries_baseline = np.copy(pred_anomalies_timeseries)
                pred_probabilities_timeseries_baseline = np.copy(pred_anomalies_timeseries)
                _, plot_ind, anomaly_ind = np.intersect1d(plot_dates, curr_anomaly_dates_baseline, return_indices=True)
                pred_anomalies_timeseries_baseline[plot_ind] = curr_pred_anomaly_baseline[anomaly_ind]
                true_anomalies_timeseries_baseline[plot_ind] = curr_true_anomaly_baseline[anomaly_ind]
                pred_probabilities_timeseries_baseline[plot_ind] = curr_probabilities_baseline[anomaly_ind]
                true_anomalies_array.append(true_anomalies_timeseries_baseline)
                pred_anomalies_array.append(pred_anomalies_timeseries_baseline)
                probabilities_array.append(pred_probabilities_timeseries_baseline)
                
            classified_timeseries_figure_comparison(sensor_id, sensor_feature_timeseries, plot_dates,
                        true_anomalies_array, pred_anomalies_array,model_config, out_dir,
                    probabilities=probabilities_array,labels=labels, showfig=True)
            
