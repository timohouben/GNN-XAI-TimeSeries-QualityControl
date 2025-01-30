import os, glob
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import xarray as xr
from libs.preprocessing_functions import wrapper_batch_mapping_plot_cml, wrapper_batch_mapping_cml, wrapper_batch_mapping_baseline_cml,\
wrapper_batch_mapping_baseline_plot_cml, create_batched_dataset, interpolate_features
from matplotlib.lines import Line2D
from sklearn.metrics import auc
from scipy import interpolate
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import Patch


def plot_roc_curves(fpr, tpr, model_config, thresholds=None, choosen_thresholds=None,
                   outpath=None, labels=['GCN', 'baseline']):
    color_tab = ['indianred', 'teal']

    if not os.path.exists(model_config.plotting.outdir):
        os.makedirs(model_config.plotting.outdir)
        
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


def extract_target_info(dataset, anomaly_date_ind, ds_type='cml', timeseries_out=False, baseline=False):
    anomaly_dates = []
    anomaly_flags_true = []
    sensor_ids = []
    anomaly_timeseries = []
    dates_timeseries = []
    for x in dataset:
        if ds_type=='cml':
            anomaly_dates.append(pd.to_datetime([curr_date.decode('utf-8') for curr_date in x[0][-1].numpy()[:, anomaly_date_ind]],
                                                format='%Y-%m-%dT%H:%M:%S'))
            if timeseries_out:
                if baseline == True:
                    anomaly_timeseries.append(x[0][0])
                else:
                    anomaly_timeseries.append(x[0][1])
                dates_timeseries.append(x[0][-1])
                    
        elif ds_type=='soilnet':
            node_numb = x[0][-3]
            anomaly_dates.append(np.repeat(pd.to_datetime([curr_date.decode('utf-8') for curr_date in x[0][-1].numpy()[:, anomaly_date_ind]],
                                            format='%Y-%m-%dT%H:%M:%S'), node_numb))
            if timeseries_out:
                sensor_dates = np.repeat(x[0][-1].numpy(), node_numb, axis=0)
                dates_timeseries.append(sensor_dates)
                sensor_anomaly_timeseries = np.reshape(x[0][0], (-1, sensor_dates.shape[1], x[0][0].shape[-1]))
                anomaly_timeseries.append(sensor_anomaly_timeseries)
                
        try:
            sensor_ids.append([sensor_id.decode('utf-8') for sensor_id in x[0][-2].numpy()])
        except AttributeError: 
            sensor_ids.append([sensor_id for sensor_id in x[0][-2].numpy()])
        anomaly_flags_true.append(x[1].numpy())
        
    anomaly_flags_true = np.concatenate(anomaly_flags_true)
    sensor_ids = np.concatenate(sensor_ids)
    anomaly_dates = np.concatenate(anomaly_dates)
    
    if timeseries_out:
        anomaly_timeseries = np.concatenate(anomaly_timeseries)
        dates_timeseries = np.concatenate(dates_timeseries)
        return sensor_ids, anomaly_dates, anomaly_flags_true, anomaly_timeseries, dates_timeseries
    else:   
        return sensor_ids, anomaly_dates, anomaly_flags_true


def timeseries_figure(predicted, true, sensor_timeseries, sensor_id, dates, outdir, anomaly_time_ind, model_config, ds_type, showfig=True):
    line_colors = ['teal', 'deepskyblue']
    alpha = model_config.plotting.alpha
    fig, ax = plt.subplots(1, 1, figsize=(18, 3))
    # select minimum value
    ymin = np.floor(np.nanmin(sensor_timeseries))
    ymax = np.ceil(np.nanmax(sensor_timeseries))
    ax.set_ylim([ymin, ymax])
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
    legend_elements = [mpatches.Patch(color='green', label='True Positive', alpha=alpha),
                       mpatches.Patch(color='blue', label='True Negative', alpha=alpha),
                   mpatches.Patch(color='orange', label='False Positive', alpha=alpha),        
                   mpatches.Patch(color='red', label='False Negative', alpha=alpha)]
    curr_date = dates[anomaly_time_ind]
    if ds_type == 'cml':
        ax.plot(dates, sensor_timeseries[:, 0], color=line_colors[0])
        ax.plot(dates, sensor_timeseries[:, 1], color=line_colors[1])
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%y-%m-%d %H:%M'))
        ax_label = 'TL [dB]'
        label_color = 'black'
    if ds_type == 'soilnet':
        ax.plot(dates, sensor_timeseries[:, 0], color=line_colors[0])
        ax2 = ax.twinx()
        ax2.plot(dates, sensor_timeseries[:, 1], color=line_colors[1])
        ax2.locator_params(axis='y', nbins=5)
        ax_label1 = 'Battery voltage normalized'
        ax2.set_ylabel(ax_label1, color=line_colors[1], fontsize=14)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H:%M'))
        label_color = line_colors[0]
        ax_label = 'Soil moisture normalized'

    ax.set_ylabel(ax_label, color=label_color, fontsize=14)
    ax.locator_params(axis='y', nbins=4)
    ax.set_title('{} on {}'.format(sensor_id, curr_date), pad=12)
    ax.legend(handles=legend_elements)
    outpath = os.path.join(outdir, '{}_{}_true_{}_pred_{}'.format(sensor_id, curr_date, true, predicted))
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.margins(0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(outpath, bbox_inches='tight')
    plt.show()
    #plt.close()
    

def plot_classified_samples(model, val_dataset, model_config, preproc_config, threshold=0.5, baseline=False, plot_example=False):
    if baseline == True:
        out_dir = os.path.join(model_config.plotting.outdir, 'classified_validation_samples_baseline')
    else:
        out_dir = os.path.join(model_config.plotting.outdir, 'classified_validation_samples')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if preproc_config.ds_type == 'soilnet':
        anomaly_date_ind = int(model.model_info[0].numpy()/model.model_info[-1].numpy())
    else:
        anomaly_date_ind = model.model_info[0].numpy()
    call_numb = tf.data.AUTOTUNE 
    val_dataset_test, _, wrapping_functions = create_batched_dataset(val_dataset, preproc_config, shuffle=False, baseline=baseline)
    anomaly_flags_pred = model.predict(val_dataset_test.map(wrapping_functions[0], call_numb).prefetch(call_numb)).flatten()
    sensor_ids_val, anomaly_dates_val, anomaly_flags_true_val, sensor_anomaly_timeseries, dates_timeseries = extract_target_info(val_dataset_test.map(
                wrapping_functions[1], call_numb).prefetch(call_numb), anomaly_date_ind, ds_type=preproc_config.ds_type,
                                                                                    timeseries_out=True, baseline=baseline)
    anomaly_flags_pred_class = np.where(anomaly_flags_pred > threshold, 1, 0)
    for b in range(anomaly_flags_pred_class.shape[0]):
        if (plot_example==True) and (b>2):
            break
        curr_dates = pd.to_datetime([x.decode('utf-8') for x in tf.squeeze(dates_timeseries[b]).numpy()], format='%Y-%m-%dT%H:%M:%S')
        curr_sensor_id = sensor_ids_val[b]
        timeseries_figure(anomaly_flags_pred_class[b], anomaly_flags_true_val[b], sensor_anomaly_timeseries[b],
                                      curr_sensor_id, curr_dates, out_dir, anomaly_date_ind, model_config, ds_type=preproc_config.ds_type)

         
def plot_results(sensor_ids, anomaly_dates, anomaly_flags_pred, anomaly_flags_true, predictions,
                 preproc_config, model_config, comparison=False, plot_neighbors=False, 
                 sensor_ids_baseline=None, anomaly_dates_baseline=None,
                 anomaly_flags_pred_baseline=None, anomaly_flags_true_baseline=None, predictions_baseline=None,
                labels=['GCN', 'baseline'], interval=None, plot_example=False):
    line_colors = ['teal', 'deepskyblue']
    color_label = 'black'
    empty_color = 'grey'
    tn_color = 'white'
    timestep_before = preproc_config.timestep_before
    timestep_after = preproc_config.timestep_after
    alpha = model_config.plotting.alpha
    if interval is None:
        interval = model_config.plotting.plot_time_range
    if comparison:
        out_dir = os.path.join(model_config.plotting.outdir, 'classified_timeseries_comparison')
    else:
        out_dir = os.path.join(model_config.plotting.outdir, 'classified_timeseries')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if preproc_config.ds_type == 'soilnet':
        ds_soilnet = xr.open_dataset(preproc_config.raw_dataset_path).load()
        ds_soilnet = ds_soilnet.dropna(dim='sensor_id', subset=['latitude', 'longitude'])
        # interpolate features
        features_list = ['moisture', 'temp', 'battv']
        interpolation_max_time = '60min'
        # interpolate data if desired
        ds = interpolate_features(ds_soilnet, features=features_list, interpolation_max=interpolation_max_time, method="linear")
        target = xr.where((ds.moisture_flag_OK==True) & (ds.moisture>0) & (ds.moisture < 100), 0, np.nan).values
        mask = (ds.moisture_flag_Manual==True) & (ds.moisture>0) & (ds.moisture < 100)
        target[mask] = 1
        ds['target'] = ('sensor_id', 'time'), target
        automatic_flags = ds.moisture_flag_no_label.where(ds.moisture_flag_no_label==False).values
        automatic_flags[(ds['moisture_flag_Auto:BattV']==True) | (ds['moisture_flag_Auto:Range']==True) |
                     (ds['moisture_flag_Auto:Spike']==True)] = True
        ds['automatic_flags'] = ('sensor_id', 'time'), automatic_flags

    counter = 0
    for sensor_id in np.unique(sensor_ids):
        if counter>4:
            break
        # select dates for the sensors being processed
        ind_sensor = np.where(sensor_ids==sensor_id)
        dates_sensor = anomaly_dates[ind_sensor]
        end_date = np.max(dates_sensor) + pd.Timedelta(minutes=timestep_after)
        start_date = np.min(dates_sensor) - pd.Timedelta(minutes=timestep_before)
        time_ranges = pd.date_range(start=start_date, end=end_date+ pd.Timedelta(hours=interval), freq='%dH' % interval)
        if preproc_config.ds_type == 'soilnet':
            ds_sensor = ds.sel(sensor_id=sensor_id)
        elif preproc_config.ds_type == 'cml':    
            data_path = os.path.join(preproc_config.ncfiles_dir, '{}.nc'.format(sensor_id))
            ds_sensor = xr.open_dataset(data_path)
        if comparison:
            ind_sensor_baseline = np.where(sensor_ids_baseline==sensor_id)
        for i in np.arange(len(time_ranges)-1):
            if counter>4:
                break
            ds_sensor_date_range = ds_sensor.sel(time=slice(time_ranges[i], time_ranges[i+1]))
            outpath = os.path.join(out_dir, '{}_{}_{}.png'.format(sensor_id, time_ranges[i], time_ranges[i+1]))
            if preproc_config.ds_type == 'cml':
                missing_data_sensor_id = np.where(np.isnan(ds_sensor_date_range.TL_1).all(axis=1) &
                                       np.isnan(ds_sensor_date_range.TL_2).all(axis=1))[0]
            else:
                missing_data_sensor_id = []
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
                        
            if comparison:
                pred_anomalies_timeseries_baseline = np.copy(pred_anomalies_timeseries)
                true_anomalies_timeseries_baseline = np.copy(pred_anomalies_timeseries)
                pred_probabilities_timeseries_baseline = np.copy(pred_anomalies_timeseries) 
            
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
                ax_label = 'TL [dB]'
                ymin = np.nanmin(sensor_feature_timeseries) - 1
                ymax = np.nanmax(sensor_feature_timeseries) + 1
            elif preproc_config.ds_type == 'soilnet':
                sensor_feature_timeseries = [ds_sensor_date_range.moisture.values.flatten()]
                all_features_timeseries = y = np.expand_dims(ds_sensor_date_range.moisture.values, -1)
                sensor_feature_timeseries1 = np.expand_dims(ds_sensor_date_range.battv.values/1000, -1)
                ymin = int(np.floor(np.nanmin(sensor_feature_timeseries)))
                ymax = int(np.ceil(np.min([60, np.nanmax(sensor_feature_timeseries)])))
                ymin1 = int(np.floor(np.nanmin(sensor_feature_timeseries1)))
                ymax1 = int(np.ceil(np.nanmax(sensor_feature_timeseries1)))            
                ax_label = 'Soil moisture [%]'
                ax_label1 = 'Battery voltage [V]'
            
            if comparison:
                base = 0.5
                fig, ax = plt.subplots(2, 1, sharex='all', height_ratios=[1.2, 1], figsize=(18, 3+3))
                ind_dates_baseline = np.where((anomaly_dates_baseline>=time_ranges[i]) &
                                              (anomaly_dates_baseline<=time_ranges[i+1]))
                curr_ind_baseline = np.intersect1d(ind_sensor_baseline, ind_dates_baseline)
                curr_pred_anomaly_baseline = anomaly_flags_pred_baseline[curr_ind_baseline]
                curr_true_anomaly_baseline = anomaly_flags_true_baseline[curr_ind_baseline]
                curr_probabilities_baseline = predictions_baseline[curr_ind_baseline]
                curr_anomaly_dates_baseline = anomaly_dates_baseline[curr_ind_baseline]
        
                _, plot_ind, anomaly_ind = np.intersect1d(plot_dates, curr_anomaly_dates_baseline, return_indices=True)
                pred_anomalies_timeseries_baseline[plot_ind] = curr_pred_anomaly_baseline[anomaly_ind]
                true_anomalies_timeseries_baseline[plot_ind] = curr_true_anomaly_baseline[anomaly_ind]
                pred_probabilities_timeseries_baseline[plot_ind] = curr_probabilities_baseline[anomaly_ind]
            else:
                base = 0
                fig, ax = plt.subplots(2, 1, sharex='all', height_ratios=[2, 1], figsize=(18, 3+1.5))
                
            curr_ax = ax[0]        
            for j, sensor_features in enumerate(sensor_feature_timeseries):
                curr_ax.plot(plot_dates, sensor_features, linewidth=2, color=line_colors[j])
                
            if preproc_config.ds_type == 'soilnet':
                color_label = line_colors[0]
                ax2 = curr_ax.twinx()
                ax2.plot(plot_dates, sensor_feature_timeseries1, linewidth='2', color=line_colors[1], zorder=1)
                ax2.set_ylabel(ax_label1, color=line_colors[1], fontsize=14)
                ax2.locator_params(axis='y', nbins=4)
                ax2.tick_params(axis='y', labelsize=14)
                day_interval = int(np.ceil(interval/(24*3*2)))
                curr_ax.xaxis.set_minor_locator(mdates.DayLocator(interval=day_interval))
                curr_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                #curr_ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 4,7, 10, 13, 16, 19, 22, 25, 28, 31]))
                curr_ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            else:
                curr_ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
                curr_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                curr_ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
            curr_ax.set_ylim([ymin, ymax])   
            curr_ax.margins(0)
            curr_ax.tick_params(axis='x', labelsize=14)
            curr_ax.tick_params(axis='y', labelsize=14)
            curr_ax.locator_params(axis='y', nbins=4)
            curr_ax.set_ylabel(ax_label, color=color_label, fontsize=14)
            curr_ax = ax[1]
            curr_ax.fill_between(plot_dates, base, 1,
                                    where=np.logical_and(pred_anomalies_timeseries==1, true_anomalies_timeseries==1),
                                     label='True Positive', alpha=alpha, color='green')
            curr_ax.fill_between(plot_dates,  base, 1,
                                    where=np.logical_and(pred_anomalies_timeseries==0, true_anomalies_timeseries==0),
                                     label='True Negative', alpha=alpha, color=tn_color)
            curr_ax.fill_between(plot_dates, base, 1,
                                    where=np.logical_and(pred_anomalies_timeseries==0, true_anomalies_timeseries==1),
                                     label='False Negative', alpha=alpha, color='red')
            curr_ax.fill_between(plot_dates, base, 1,
                                    where=np.logical_and(pred_anomalies_timeseries==1, true_anomalies_timeseries==0),
                                     label='False Positive', alpha=alpha, color='orange')
            if preproc_config.ds_type == 'soilnet':
                # Automatic flags
                curr_ax.fill_between(plot_dates, base, 1,
                                    where=(ds_sensor_date_range.automatic_flags.values),
                                     label='Automatic flag', alpha=alpha, color='blue')
                # No data 
                curr_ax.fill_between(plot_dates, base, 1,
                                    where=np.logical_and(np.isnan(true_anomalies_timeseries)==1, ds_sensor_date_range.automatic_flags.values==0),
                                     label='No data', alpha=alpha, color=empty_color)
            else:
                # No data 
                curr_ax.fill_between(plot_dates, base, 1,
                                        where=np.isnan(true_anomalies_timeseries),
                                         label='No data', alpha=alpha, color=empty_color)

            if comparison:
                curr_ax.fill_between(plot_dates, 0, 0.5,
                                        where=np.logical_and(pred_anomalies_timeseries_baseline==1, true_anomalies_timeseries_baseline==1),
                                        alpha=alpha, color='green')
                curr_ax.fill_between(plot_dates, 0, 0.5,
                                        where=np.logical_and(pred_anomalies_timeseries_baseline==0, true_anomalies_timeseries_baseline==0),
                                        alpha=alpha, color=tn_color)
                curr_ax.fill_between(plot_dates, 0, 0.5,
                                        where=np.logical_and(pred_anomalies_timeseries_baseline==0, true_anomalies_timeseries_baseline==1),
                                        alpha=alpha, color='red')
                curr_ax.fill_between(plot_dates, 0, 0.5,
                                        where=np.logical_and(pred_anomalies_timeseries_baseline==1, true_anomalies_timeseries_baseline==0),
                                        alpha=alpha, color='orange')
                
                if preproc_config.ds_type == 'soilnet':
                    # Automatic flags
                    curr_ax.fill_between(plot_dates, 0, 0.5,
                                        where=(ds_sensor_date_range.automatic_flags.values),
                                        alpha=alpha, color='blue')
                    # No data 
                    curr_ax.fill_between(plot_dates, 0, 0.5,
                                        where=np.logical_and(np.isnan(true_anomalies_timeseries)==1,
                                                             ds_sensor_date_range.automatic_flags.values==0),
                                        alpha=alpha, color=empty_color)
                else:
                    # No data 
                    curr_ax.fill_between(plot_dates, 0, 0.5,
                                            where=np.isnan(true_anomalies_timeseries),
                                            alpha=alpha, color=empty_color)
                curr_ax.axhline(0.5, color='black', alpha=alpha)
                curr_ax.text(-0.05, 0.25, 'baseline', transform=curr_ax.transAxes, fontsize=12) 
            
            ax[0].tick_params(labelbottom=True)
            handles, legend_labels = curr_ax.get_legend_handles_labels()
            curr_ax.text(-0.05, 0.5 + base/2, 'GCN', transform=curr_ax.transAxes, fontsize=12) 
            curr_ax.set_axis_off()            
            new_handles = []
            for h in range(len(handles)):
                if h==1:
                    edgecolor = [0,  0,  0,  alpha]
                else:
                    edgecolor = handles[h].get_edgecolor()
                color = handles[h].get_facecolor()
                new_handles.append(Patch(facecolor=color, edgecolor=edgecolor, label=legend_labels[h]))
            curr_ax.legend(handles=new_handles, loc=10, bbox_to_anchor=(0.5, -0.1), ncols=6)
            counter = counter + 1
            fig.suptitle('{}'.format(sensor_id), y=0.99)
            #plt.tight_layout()
            plt.tight_layout(pad=0, h_pad=1.08, w_pad=0)
            plt.margins(0)    
            plt.savefig(outpath, bbox_inches='tight')
            plt.show()
