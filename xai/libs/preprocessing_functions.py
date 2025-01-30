import geopy.distance, glob, os, random, spektral, shutil
import scipy.sparse as sp
import xarray as xr
import pandas as pd
import numpy as np
import tensorflow as tf
from inspect import signature
from functools import reduce


def calculate_statistics(ds, preproc_config):
    if preproc_config.ds_type == 'cml':
        ds['TL_1_mean'] = ds['TL_1'].mean("time")
        ds['TL_2_mean'] = ds['TL_2'].mean("time")
        ds['TL_1_std'] = ds['TL_1'].std("time")
        ds['TL_2_std'] = ds['TL_2'].std("time")
        ds['TL_1_min'] = ds['TL_1'].min("time")
        ds['TL_2_min'] = ds['TL_2'].min("time")
        ds['TL_1_max'] = ds['TL_1'].max("time")
        ds['TL_2_max'] = ds['TL_2'].max("time") 
        ds['TL_1_median'] = ds['TL_1'].median("time")
        ds['TL_2_median'] = ds['TL_2'].median("time")
        ds['TL_1_rolling_mean'] = ds['TL_1'].rolling(time=preproc_config.dataset.window_length, min_periods=1).mean()
        ds['TL_2_rolling_mean'] = ds['TL_2'].rolling(time=preproc_config.dataset.window_length, min_periods=1).mean()
        ds['TL_1_rolling_std'] = ds['TL_1'].rolling(time=preproc_config.dataset.window_length, min_periods=1).std()
        ds['TL_2_rolling_std'] = ds['TL_2'].rolling(time=preproc_config.dataset.window_length, min_periods=1).std()
        ds['TL_1_rolling_median'] = ds['TL_1'].rolling(time=preproc_config.dataset.window_length, min_periods=1).median()
        ds['TL_2_rolling_median'] = ds['TL_2'].rolling(time=preproc_config.dataset.window_length, min_periods=1).median()
    elif preproc_config.ds_type == 'soilnet':
        ds['moisture_mean'] = ds['moisture'].mean("time")
        ds['battv_mean'] = ds['battv'].mean("time")
        ds['temp_mean'] = ds['temp'].mean("time")
    
        ds['moisture_std'] = ds['moisture'].std("time")
        ds['battv_std'] = ds['battv'].std("time")
        ds['temp_std'] = ds['temp'].std("time")
        
        ds['moisture_min'] = ds['moisture'].min("time")
        ds['battv_min'] = ds['battv'].min("time")
        ds['temp_min'] = ds['temp'].min("time")
        
        ds['moisture_max'] = ds['moisture'].max("time")
        ds['battv_max'] = ds['battv'].max("time")
        ds['temp_max'] = ds['temp'].max("time") 
        
        ds['moisture_median'] = ds['moisture'].median("time")
        ds['battv_median'] = ds['battv'].median("time")
        ds['temp_median'] = ds['temp'].median("time")    
    
        ds['moisture_rolling_mean'] = ds['moisture'].rolling(time=preproc_config.dataset.window_length, min_periods=1).mean()
        ds['battv_rolling_mean'] = ds['battv'].rolling(time=preproc_config.dataset.window_length, min_periods=1).mean()
        ds['temp_rolling_mean'] = ds['temp'].rolling(time=preproc_config.dataset.window_length, min_periods=1).mean()
    
        ds['moisture_rolling_std'] = ds['moisture'].rolling(time=preproc_config.dataset.window_length, min_periods=1).std()
        ds['battv_rolling_std'] = ds['battv'].rolling(time=preproc_config.dataset.window_length, min_periods=1).std()
        ds['temp_rolling_std'] = ds['temp'].rolling(time=preproc_config.dataset.window_length, min_periods=1).std() 
        
        ds['moisture_rolling_median'] = ds['moisture'].rolling(time=preproc_config.dataset.window_length, min_periods=1).median()
        ds['battv_rolling_median'] = ds['battv'].rolling(time=preproc_config.dataset.window_length, min_periods=1).median()
        ds['temp_rolling_median'] = ds['temp'].rolling(time=preproc_config.dataset.window_length, min_periods=1).median()
    return ds


def create_tfrecords_dataset(preproc_config, min_date=None, max_date=None):
    max_distance = preproc_config.graph.max_neighbour_distance
    timestep_before = preproc_config.timestep_before
    timestep_after = preproc_config.timestep_after
    if preproc_config.ds_type == 'cml':
        freq = 1
    elif preproc_config.ds_type == 'soilnet':
        freq = 15
        max_depth = preproc_config.graph.max_depth
    sequence_length = int((timestep_after + timestep_before)/freq + 1)
    num_samples = 2_000
    tfrecords_dir = os.path.join(preproc_config.dataset.tfrecords_dataset_dir,
                            '{}_{}'.format(timestep_before, timestep_after))
    if os.path.exists(tfrecords_dir):
        print('Dir removed: {}'.format(tfrecords_dir))
        shutil.rmtree(tfrecords_dir)
    os.makedirs(tfrecords_dir)
    netcdf_files = np.array(glob.glob(os.path.join(preproc_config.dataset.ncfiles_dir, '*.nc')))
    for number, netcdf_file in enumerate(netcdf_files):
        sensor_with_neighbours_ds = xr.open_dataset(netcdf_file)
        # Calculate statistics for normalization
        sensor_with_neighbours_ds = calculate_statistics(sensor_with_neighbours_ds, preproc_config)
        print('Statistics calculated')
        anomalous_sensor_id = sensor_with_neighbours_ds.sensor_id[sensor_with_neighbours_ds.flagged].values[0]
        print('Processing %s sensor: %s' % (preproc_config.ds_type, anomalous_sensor_id))
        
        if (min_date is None) and (max_date is None):
            dates_processing = sensor_with_neighbours_ds.time.values
        elif min_date is None:
            dates_processing = sensor_with_neighbours_ds.time.values[sensor_with_neighbours_ds.time<=max_date]
        elif max_date is None:
            dates_processing = sensor_with_neighbours_ds.time.values[sensor_with_neighbours_ds.time>=min_date]
        else:
            dates_processing = sensor_with_neighbours_ds.time.values[(sensor_with_neighbours_ds.time>=min_date) &
            (sensor_with_neighbours_ds.time<=max_date)]
            
        num_tfrecords =  len(dates_processing) // num_samples
        if len(dates_processing) % num_samples:
            num_tfrecords += 1  # add one record if there are any remaining samples
        print('Files number: ', num_tfrecords)
        for tfrec_num in range(num_tfrecords):
            sample_dates = dates_processing[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)]
            print('File numb: %d' % (tfrec_num))
            with tf.io.TFRecordWriter(
                tfrecords_dir + "/%s_%03d.tfrec" % (anomalous_sensor_id, tfrec_num)
                        ) as writer:    
                for i, processed_date in enumerate(sample_dates):
                    date_after = processed_date + pd.Timedelta(minutes=timestep_after)
                    date_before = processed_date - pd.Timedelta(minutes=timestep_before)
                    sample_ds = sensor_with_neighbours_ds.sel(time=slice(date_before, date_after))
                    if len(sample_ds.time.values)<sequence_length:
                        continue
                    if preproc_config.ds_type == 'cml':
                        if (np.isnan(sample_ds.TL_1[sample_ds.flagged][0]).any().values) or (np.isnan(sample_ds.TL_2[sample_ds.flagged][0]).any().values):
                            continue
                        missing_data_id = np.union1d(np.where(np.isnan(sample_ds.TL_1).any(axis=1)),
                          np.where(np.isnan(sample_ds.TL_2).any(axis=1)))
                    elif preproc_config.ds_type == 'soilnet':
                        if (np.isnan(sample_ds.moisture[sample_ds.flagged][0]).any().values) or (
                            np.isnan(sample_ds.temp[sample_ds.flagged][0]).any().values) or (
                            np.isnan(sample_ds.battv[sample_ds.flagged][0]).any().values):
                                continue
                        # check if the any time series has gaps
                        missing_data_id = reduce(np.union1d,
                                (np.where(np.isnan(sample_ds.moisture).any(axis=1)),
                                np.where(np.isnan(sample_ds.temp).any(axis=1)),
                                np.where(np.isnan(sample_ds.battv).any(axis=1))))
                    if len(missing_data_id) > 0:
                        sample_ds = sample_ds.drop_isel(sensor_id=missing_data_id)
                    if preproc_config.ds_type == 'cml':
                        adjacency_matrix = sample_ds.distances<max_distance
                    elif preproc_config.ds_type == 'soilnet':
                        if sample_ds.moisture.values.shape[0] == 0:
                            continue
                        adjacency_matrix = (((sample_ds.distances.values<=max_distance) & (sample_ds.depths.values==0))
                                            | ((sample_ds.distances.values==0) & (sample_ds.depths.values<=max_depth)))
 
                    # Create the SequenceExample
                    example = create_example(processed_date, sample_ds, anomalous_sensor_id, adjacency_matrix, sequence_length, preproc_config.ds_type)
                    writer.write(example.SerializeToString())


def create_example(processed_date, sample_ds, anomalous_sensor_id, adjacency_matrix, sequence_length, ds_type='cml'):
    nodes, neighbours = np.where(adjacency_matrix)
    distances = sample_ds.distances.values[adjacency_matrix]
    dates_sample = sample_ds.time.values
    anomalous_date = processed_date
    anomaly_flag_sample = int(sample_ds.target.sel(time=processed_date).values*1)
    dates_sample_str = np.datetime_as_string(dates_sample, unit='s')
    sample_anomaly_date_ds = sample_ds.sel(time=pd.to_datetime(processed_date))
    sensor_ids = sample_ds.sensor_id.values
    if ds_type == 'cml':
        trsl1 = sample_ds.TL_1.values
        trsl2 = sample_ds.TL_2.values
        trsl1_anomalous_cml = sample_ds.TL_1[sample_ds.flagged].values
        trsl2_anomalous_cml = sample_ds.TL_2[sample_ds.flagged].values
        cml_lat_a = sample_ds.site_a_latitude.values
        cml_lon_a = sample_ds.site_a_longitude.values
        cml_lat_b = sample_ds.site_b_latitude.values
        cml_lon_b = sample_ds.site_b_longitude.values
        cml_ids = sample_ds.sensor_id.values
        context = tf.train.Features(feature={ 
            'anomaly_ID': bytes_feature(anomalous_sensor_id),
            'TRSL1_anomalous_cml': float_feature_from_list(trsl1_anomalous_cml.flatten()),
            'TRSL2_anomalous_cml': float_feature_from_list(trsl2_anomalous_cml.flatten()),
            'TRSL1_mean': float_feature_from_list(sample_ds.TL_1_mean.values.flatten()),
            'TRSL2_mean': float_feature_from_list(sample_ds.TL_2_mean.values.flatten()),
            'TRSL1_median': float_feature_from_list(sample_ds.TL_1_median.values.flatten()),
            'TRSL2_median': float_feature_from_list(sample_ds.TL_2_median.values.flatten()),
            'TRSL1_std': float_feature_from_list(sample_ds.TL_1_std.values.flatten()),
            'TRSL2_std': float_feature_from_list(sample_ds.TL_2_std.values.flatten()),
            'TRSL1_min': float_feature_from_list(sample_ds.TL_1_min.values.flatten()),
            'TRSL2_min': float_feature_from_list(sample_ds.TL_2_min.values.flatten()),
            'TRSL1_max': float_feature_from_list(sample_ds.TL_1_max.values.flatten()),
            'TRSL2_max': float_feature_from_list(sample_ds.TL_2_max.values.flatten()),
            'TRSL1_rolling_mean': float_feature_from_list(
                sample_anomaly_date_ds.TL_1_rolling_mean.values.flatten()),
            'TRSL2_rolling_mean': float_feature_from_list(
                sample_anomaly_date_ds.TL_2_rolling_mean.values.flatten()),
            'TRSL1_rolling_std': float_feature_from_list(
                sample_anomaly_date_ds.TL_1_rolling_std.values.flatten()),
            'TRSL2_rolling_std': float_feature_from_list(
                sample_anomaly_date_ds.TL_2_rolling_std.values.flatten()),
            'TRSL1_rolling_median': float_feature_from_list(
                sample_anomaly_date_ds.TL_1_rolling_median.values.flatten()),
            'TRSL2_rolling_median': float_feature_from_list(
                sample_anomaly_date_ds.TL_2_rolling_median.values.flatten()),    
            'anomaly_flag': int64_feature(anomaly_flag_sample),
            'node_numb': int64_feature(trsl1.shape[0]),
            'link_numb': int64_feature(len(nodes)),
            'CML_ids': bytes_feature_from_list(cml_ids),
            'dates': bytes_feature_from_list(dates_sample_str),
                    })
        
        timeseries = tf.train.FeatureLists(feature_list={
            'TRSL1': float_featurelist_from_list(trsl1.T),
            'TRSL2': float_featurelist_from_list(trsl2.T),
            'nodes': int64_featurelist(nodes),
            'neighbours': int64_featurelist(neighbours),
            'distances': float_featurelist(distances),
            'cml_lat_a': coordinates_featurelist(cml_lat_a, sequence_length),
            'cml_lat_b': coordinates_featurelist(cml_lat_b, sequence_length),
            'cml_lon_a': coordinates_featurelist(cml_lon_a, sequence_length),
            'cml_lon_b': coordinates_featurelist(cml_lon_b, sequence_length),
                             })
        
    elif ds_type == 'soilnet':
        depths = sample_ds.depths.values[adjacency_matrix]
        sensor_lat = sample_ds.latitude.values
        sensor_lon = sample_ds.longitude.values
        moisture = sample_ds.moisture.values
        battv = sample_ds.battv.values
        anomaly_flags = sample_ds.target.values
        temp = sample_ds.temp.values
        moisture_anomalous_sensor = sample_ds.moisture[sample_ds.flagged].values
        temp_anomalous_sensor = sample_ds.temp[sample_ds.flagged].values 
        battv_anomalous_sensor = sample_ds.battv[sample_ds.flagged].values
        
        context = tf.train.Features(feature={
            'anomaly_ID': int64_feature(anomalous_sensor_id), 
            'sensor_ids': int64_feature_from_list(sensor_ids), 
            'moisture_anomalous_sensor': float_feature_from_list(moisture_anomalous_sensor.flatten()),
            'temp_anomalous_sensor': float_feature_from_list(temp_anomalous_sensor.flatten()),
            'battv_anomalous_sensor': float_feature_from_list(battv_anomalous_sensor.flatten()),
             'moisture_mean': float_feature_from_list(sample_ds.moisture_mean.values.flatten()),
            'temp_mean': float_feature_from_list(sample_ds.temp_mean.values.flatten()),
            'battv_mean': float_feature_from_list(sample_ds.battv_mean.values.flatten()),
            'moisture_median': float_feature_from_list(sample_ds.moisture_median.values.flatten()),
            'temp_median': float_feature_from_list(sample_ds.temp_median.values.flatten()),
            'battv_median': float_feature_from_list(sample_ds.battv_median.values.flatten()),
            'moisture_std': float_feature_from_list(sample_ds.moisture_std.values.flatten()),
            'temp_std': float_feature_from_list(sample_ds.temp_std.values.flatten()),
            'battv_std': float_feature_from_list(sample_ds.battv_std.values.flatten()),
            'moisture_min': float_feature_from_list(sample_ds.moisture_min.values.flatten()),
            'temp_min': float_feature_from_list(sample_ds.temp_min.values.flatten()),
            'battv_min': float_feature_from_list(sample_ds.battv_min.values.flatten()),
            'moisture_max': float_feature_from_list(sample_ds.moisture_max.values.flatten()),
            'temp_max': float_feature_from_list(sample_ds.temp_max.values.flatten()),
            'battv_max': float_feature_from_list(sample_ds.battv_max.values.flatten()),    
            'moisture_rolling_mean': float_feature_from_list(sample_anomaly_date_ds.moisture_rolling_mean.values.flatten()),
            'temp_rolling_mean': float_feature_from_list(sample_anomaly_date_ds.temp_rolling_mean.values.flatten()),
            'battv_rolling_mean': float_feature_from_list(sample_anomaly_date_ds.battv_rolling_mean.values.flatten()),      
            'moisture_rolling_std': float_feature_from_list(sample_anomaly_date_ds.moisture_rolling_std.values.flatten()),
            'temp_rolling_std': float_feature_from_list(sample_anomaly_date_ds.temp_rolling_std.values.flatten()),
            'battv_rolling_std': float_feature_from_list(sample_anomaly_date_ds.battv_rolling_std.values.flatten()),
            'moisture_rolling_median': float_feature_from_list(sample_anomaly_date_ds.moisture_rolling_median.values.flatten()),
            'temp_rolling_median': float_feature_from_list(sample_anomaly_date_ds.temp_rolling_median.values.flatten()),
            'battv_rolling_median': float_feature_from_list(sample_anomaly_date_ds.battv_rolling_median.values.flatten()),
            'anomaly_flag': int64_feature(anomaly_flag_sample),
            'node_numb': int64_feature(moisture.shape[0]),
            'link_numb': int64_feature(len(nodes)),
            'dates': bytes_feature_from_list(dates_sample_str),
            })
        
        timeseries = tf.train.FeatureLists(feature_list={
                'moisture': float_featurelist_from_list(moisture.T),
                'temp': float_featurelist_from_list(temp.T),
                'battv': float_featurelist_from_list(battv.T),
                'nodes': int64_featurelist(nodes),
                'neighbours': int64_featurelist(neighbours),
                'distances': float_featurelist(distances),
                'depths': float_featurelist(depths),
                'sensor_lat': coordinates_featurelist(sensor_lat, sequence_length),
                'sensor_lon': coordinates_featurelist(sensor_lon, sequence_length),
                })
    # Create the SequenceExample
    example = tf.train.SequenceExample(context=context, feature_lists=timeseries)
    return example


@tf.function
def parse_tfrecord_fn_baseline(example, normalization='rolling_median'):
    feature_description = {
        'TRSL1': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2': tf.io.RaggedFeature(dtype=tf.float32),
        'cml_lat_a': tf.io.RaggedFeature(dtype=tf.float32),
        'cml_lat_b': tf.io.RaggedFeature(dtype=tf.float32),
        'cml_lon_a': tf.io.RaggedFeature(dtype=tf.float32),
        'cml_lon_b': tf.io.RaggedFeature(dtype=tf.float32),
        'nodes': tf.io.RaggedFeature(dtype=tf.int64),
        'neighbours': tf.io.RaggedFeature(dtype=tf.int64),
        'distances': tf.io.RaggedFeature(dtype=tf.float32),
    }
    context_description = {
        'anomaly_ID': tf.io.FixedLenFeature([], dtype=tf.string),
        'CML_ids': tf.io.RaggedFeature(dtype=tf.string),
        'dates': tf.io.RaggedFeature(dtype=tf.string),
        'TRSL1_anomalous_cml': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_anomalous_cml': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_median': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_median': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_std': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_std': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_min': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_min': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_max': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_max': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_rolling_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_rolling_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_rolling_std': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_rolling_std': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_rolling_median': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_rolling_median': tf.io.RaggedFeature(dtype=tf.float32),
        'anomaly_flag': tf.io.FixedLenFeature([], dtype=tf.int64),
        'node_numb': tf.io.FixedLenFeature([], dtype=tf.int64),
        'link_numb': tf.io.FixedLenFeature([], dtype=tf.int64),
    }
    example = tf.io.parse_single_sequence_example(example, context_features=context_description,
                                                  sequence_features=feature_description)
    anomaly_id = example[0]['anomaly_ID']
    cml_ind = tf.squeeze(tf.where(example[0]['CML_ids'] == anomaly_id))
    trsl1_anomaly = example[0]['TRSL1_anomalous_cml']
    trsl2_anomaly = example[0]['TRSL2_anomalous_cml']
    if normalization == 'standarization':
        trsl1_anomaly = (trsl1_anomaly - tf.gather(example[0]['TRSL1_mean'], cml_ind))/tf.gather(example[0]['TRSL1_std'], cml_ind)
        trsl2_anomaly = (trsl2_anomaly - tf.gather(example[0]['TRSL2_mean'], cml_ind))/tf.gather(example[0]['TRSL2_std'], cml_ind)
    elif normalization == 'scale':                                                              
        trsl1_anomaly = (trsl1_anomaly - tf.gather(example[0]['TRSL1_min'], cml_ind))/(
                    tf.gather(example[0]['TRSL1_max'], cml_ind) - tf.gather(example[0]['TRSL1_min'], cml_ind))
        trsl2_anomaly = (trsl2_anomaly - tf.gather(example[0]['TRSL2_min'], cml_ind))/(
                    tf.gather(example[0]['TRSL2_max'], cml_ind) - tf.gather(example[0]['TRSL2_min'], cml_ind))
    elif normalization == 'median':
        trsl1_anomaly = (trsl1_anomaly - tf.gather(example[0]['TRSL1_median'], cml_ind))/tf.gather(
            example[0]['TRSL1_median'], cml_ind)
        trsl2_anomaly = (trsl2_anomaly - tf.gather(example[0]['TRSL2_median'], cml_ind))/tf.gather(
            example[0]['TRSL2_median'], cml_ind)
    elif normalization == 'rolling_median':
        trsl1_anomaly = trsl1_anomaly - tf.gather(example[0]['TRSL1_rolling_median'], cml_ind)
        trsl2_anomaly = trsl2_anomaly - tf.gather(example[0]['TRSL2_rolling_median'], cml_ind)
    elif normalization == 'rolling_median_fractional':
        trsl1_anomaly = (trsl1_anomaly - tf.gather(example[0]['TRSL1_rolling_median'], cml_ind))/tf.gather(
            example[0]['TRSL1_rolling_median'], cml_ind)
        trsl2_anomaly = (trsl2_anomaly - tf.gather(example[0]['TRSL2_rolling_median'], cml_ind))/tf.gather(
            example[0]['TRSL2_rolling_median'], cml_ind)
    elif normalization == 'rolling_mean':
        trsl1_anomaly = (trsl1_anomaly - tf.gather(example[0]['TRSL1_rolling_mean'], cml_ind))/tf.gather(
            example[0]['TRSL1_rolling_std'], cml_ind)
        trsl2_anomaly = (trsl2_anomaly - tf.gather(example[0]['TRSL2_rolling_mean'], cml_ind))/tf.gather(
            example[0]['TRSL2_rolling_std'], cml_ind)    
    return trsl1_anomaly, trsl2_anomaly, anomaly_id, example[0]['dates'], example[0]['anomaly_flag']


@tf.function
def parse_tfrecord_fn(example, normalization='rolling_median'):
    feature_description = {
        'TRSL1': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2': tf.io.RaggedFeature(dtype=tf.float32),
        'cml_lat_a': tf.io.RaggedFeature(dtype=tf.float32),
        'cml_lat_b': tf.io.RaggedFeature(dtype=tf.float32),
        'cml_lon_a': tf.io.RaggedFeature(dtype=tf.float32),
        'cml_lon_b': tf.io.RaggedFeature(dtype=tf.float32),
        'nodes': tf.io.RaggedFeature(dtype=tf.int64),
        'neighbours': tf.io.RaggedFeature(dtype=tf.int64),
        'distances': tf.io.RaggedFeature(dtype=tf.float32),
    }
    context_description = {
        'anomaly_ID': tf.io.FixedLenFeature([], dtype=tf.string),
        'CML_ids': tf.io.RaggedFeature(dtype=tf.string),
        'dates': tf.io.RaggedFeature(dtype=tf.string),
        'TRSL1_anomalous_cml': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_anomalous_cml': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_median': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_median': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_std': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_std': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_min': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_min': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_max': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_max': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_rolling_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_rolling_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_rolling_std': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_rolling_std': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL1_rolling_median': tf.io.RaggedFeature(dtype=tf.float32),
        'TRSL2_rolling_median': tf.io.RaggedFeature(dtype=tf.float32),
        'anomaly_flag': tf.io.FixedLenFeature([], dtype=tf.int64),
        'node_numb': tf.io.FixedLenFeature([], dtype=tf.int64),
        'link_numb': tf.io.FixedLenFeature([], dtype=tf.int64),
    }
    example = tf.io.parse_single_sequence_example(example, context_features=context_description,
                                                  sequence_features=feature_description)
    anomaly_id = example[0]['anomaly_ID']
    cml_ind = tf.squeeze(tf.where(example[0]['CML_ids'] == anomaly_id))
    trsl1 = example[1]['TRSL1'].to_tensor()
    trsl2 = example[1]['TRSL2'].to_tensor()
    if normalization == 'standarization':
        trsl1 = (trsl1 - example[0]['TRSL1_mean'])/example[0]['TRSL1_std']
        trsl2 = (trsl2 - example[0]['TRSL2_mean'])/example[0]['TRSL2_std']
    elif normalization == 'scale':                                                              
        trsl1 = (trsl1 - example[0]['TRSL1_min'])/(example[0]['TRSL1_max'] - example[0]['TRSL1_min'])
        trsl2 = (trsl2 - example[0]['TRSL2_min'])/(example[0]['TRSL2_max'] - example[0]['TRSL2_min'])
    elif normalization == 'median':
        trsl1 = (trsl1 - example[0]['TRSL1_median'])/example[0]['TRSL1_median']
        trsl2 = (trsl2 - example[0]['TRSL2_median'])/example[0]['TRSL2_median']
    elif normalization == 'rolling_median':
        trsl1 = (trsl1 - example[0]['TRSL1_rolling_median'])
        trsl2 = (trsl2 - example[0]['TRSL2_rolling_median'])
    elif normalization == 'rolling_median_fractional':
        trsl1 = (trsl1 - example[0]['TRSL1_rolling_median'])/example[0]['TRSL1_rolling_median']
        trsl2 = (trsl2 - example[0]['TRSL2_rolling_median'])/example[0]['TRSL2_rolling_median']
    elif normalization == 'rolling_mean':
        trsl1 = (trsl1 - example[0]['TRSL1_rolling_mean'])/example[0]['TRSL1_rolling_std']
        trsl2 = (trsl2 - example[0]['TRSL2_rolling_mean'])/example[0]['TRSL2_rolling_std']
        
    trsl1_anomaly = tf.gather(trsl1, cml_ind, axis=-1)
    trsl2_anomaly = tf.gather(trsl2, cml_ind, axis=-1)
    return tf.RaggedTensor.from_tensor(trsl1), tf.RaggedTensor.from_tensor(trsl2), trsl1_anomaly, trsl2_anomaly,\
    example[1]['cml_lat_a'], example[1]['cml_lat_b'], example[1]['cml_lon_a'], example[1]['cml_lon_b'],\
    example[1]['distances'], example[0]['node_numb'], example[0]['link_numb'], example[1]['nodes'],\
    example[1]['neighbours'], cml_ind, anomaly_id, example[0]['dates'], example[0]['anomaly_flag']\


@tf.function
def parse_soilnet_tfrecord_fn(example, normalization='scale_range'): 
    feature_description = {
        'moisture': tf.io.RaggedFeature(dtype=tf.float32),
        'temp': tf.io.RaggedFeature(dtype=tf.float32),
        'battv': tf.io.RaggedFeature(dtype=tf.float32),
        'sensor_lat': tf.io.RaggedFeature(dtype=tf.float32),
        'sensor_lon': tf.io.RaggedFeature(dtype=tf.float32),
        'nodes': tf.io.RaggedFeature(dtype=tf.int64),
        'neighbours': tf.io.RaggedFeature(dtype=tf.int64),
        'distances': tf.io.RaggedFeature(dtype=tf.float32),
        'depths': tf.io.RaggedFeature(dtype=tf.float32),
    }
    context_description = {
        'moisture_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_median': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_median': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_median': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_std': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_std': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_std': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_min': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_min': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_min': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_max': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_max': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_max': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_rolling_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_rolling_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_rolling_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_rolling_std': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_rolling_std': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_rolling_std': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_rolling_median': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_rolling_median': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_rolling_median': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_anomalous_sensor': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_anomalous_sensor': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_anomalous_sensor': tf.io.RaggedFeature(dtype=tf.float32),
        'anomaly_ID': tf.io.FixedLenFeature([], dtype=tf.int64),
        'sensor_ids': tf.io.RaggedFeature(dtype=tf.int64),
        'anomaly_flag': tf.io.FixedLenFeature([], dtype=tf.int64),
        'node_numb': tf.io.FixedLenFeature([], dtype=tf.int64),
        'link_numb': tf.io.FixedLenFeature([], dtype=tf.int64),
        'dates': tf.io.RaggedFeature(dtype=tf.string),
    }
    example = tf.io.parse_single_sequence_example(example, context_features=context_description,
                                                  sequence_features=feature_description)
    anomaly_id = example[0]['anomaly_ID']
    moisture = example[1]['moisture'].to_tensor()
    temp = example[1]['temp'].to_tensor()
    battv = example[1]['battv'].to_tensor()
    sensor_ind = tf.squeeze(tf.where(example[0]['sensor_ids'] == anomaly_id))
    
    if normalization == 'standarization':
        moisture = (moisture - example[0]['moisture_mean'])/example[0]['moisture_std']
        temp = (temp - example[0]['temp_mean'])/example[0]['temp_std']
        battv = (battv - example[0]['battv_mean'])/example[0]['battv_std']
    elif normalization == 'scale_range':                                                         
        moisture = moisture/60
        temp_min = -20
        temp_max = 40
        temp = (temp - temp_min)/(temp_max - temp_min)
        battv_min = 2800
        battv_max = 3600
        battv = (battv - battv_min)/(battv_max - battv_min)   
    elif normalization == 'scale':                                                              
        moisture = (moisture - example[0]['moisture_min'])/(example[0]['moisture_max'] - example[0]['moisture_min'])
        temp = (temp - example[0]['temp_min'])/(example[0]['temp_max'] - example[0]['temp_min'])
        battv = (battv - example[0]['battv_min'])/(example[0]['battv_max'] - example[0]['battv_min'])
    elif normalization == 'median':
        moisture = (moisture - example[0]['moisture_median'])#/example[0]['moisture_median']
        temp = (temp - example[0]['temp_median'])#/example[0]['temp_median']
        battv = (battv - example[0]['battv_median'])#/example[0]['battv_median']
    elif normalization == 'rolling_median':
        moisture = (moisture - example[0]['moisture_rolling_median'])
        temp = (temp - example[0]['temp_rolling_median'])
        battv = (battv - example[0]['battv_rolling_median'])
    elif normalization == 'rolling_median_fractional':
        moisture = (moisture - example[0]['moisture_rolling_median'])/example[0]['moisture_rolling_median']
        temp = (temp - example[0]['temp_rolling_median'])/example[0]['temp_rolling_median']
        battv = (battv - example[0]['battv_rolling_median'])/example[0]['battv_rolling_median']
    elif normalization == 'rolling_mean':
        moisture = (moisture - example[0]['moisture_rolling_mean'])/example[0]['moisture_rolling_std']
        temp = (temp - example[0]['temp_rolling_mean'])/example[0]['temp_rolling_std']
        battv = (battv - example[0]['battv_rolling_mean'])/example[0]['battv_rolling_std']
        
    battv_anomaly = tf.gather(battv, sensor_ind, axis=-1)
    moisture_anomaly = tf.gather(moisture, sensor_ind, axis=-1)
    temp_anomaly = tf.gather(temp, sensor_ind, axis=-1)
    
    return tf.RaggedTensor.from_tensor(moisture), tf.RaggedTensor.from_tensor(temp), tf.RaggedTensor.from_tensor(battv),\
    moisture_anomaly, temp_anomaly, battv_anomaly, example[1]['sensor_lat'], example[1]['sensor_lon'],\
    example[1]['distances'], example[1]['depths'], example[0]['node_numb'], example[0]['link_numb'], \
    example[1]['nodes'], example[1]['neighbours'], sensor_ind, anomaly_id, example[0]['dates'],\
    example[0]['anomaly_flag']


@tf.function
def parse_soilnet_tfrecord_fn_baseline(example, normalization='scale_range'):
    feature_description = {
        'moisture': tf.io.RaggedFeature(dtype=tf.float32),
        'temp': tf.io.RaggedFeature(dtype=tf.float32),
        'battv': tf.io.RaggedFeature(dtype=tf.float32),
        'sensor_lat': tf.io.RaggedFeature(dtype=tf.float32),
        'sensor_lon': tf.io.RaggedFeature(dtype=tf.float32),
        'nodes': tf.io.RaggedFeature(dtype=tf.int64),
        'neighbours': tf.io.RaggedFeature(dtype=tf.int64),
        'distances': tf.io.RaggedFeature(dtype=tf.float32),
        'depths': tf.io.RaggedFeature(dtype=tf.float32),
    }
    context_description = {
        'moisture_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_median': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_median': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_median': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_std': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_std': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_std': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_min': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_min': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_min': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_max': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_max': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_max': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_rolling_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_rolling_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_rolling_mean': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_rolling_std': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_rolling_std': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_rolling_std': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_rolling_median': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_rolling_median': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_rolling_median': tf.io.RaggedFeature(dtype=tf.float32),
        'moisture_anomalous_sensor': tf.io.RaggedFeature(dtype=tf.float32),
        'temp_anomalous_sensor': tf.io.RaggedFeature(dtype=tf.float32),
        'battv_anomalous_sensor': tf.io.RaggedFeature(dtype=tf.float32),
        'anomaly_ID': tf.io.FixedLenFeature([], dtype=tf.int64),
        'sensor_ids': tf.io.RaggedFeature(dtype=tf.int64),
        'anomaly_flag': tf.io.FixedLenFeature([], dtype=tf.int64),
        'node_numb': tf.io.FixedLenFeature([], dtype=tf.int64),
        'link_numb': tf.io.FixedLenFeature([], dtype=tf.int64),
        'dates': tf.io.RaggedFeature(dtype=tf.string),#tf.io.FixedLenFeature([], dtype=tf.string),
    }
    example = tf.io.parse_single_sequence_example(example, context_features=context_description,
                                                  sequence_features=feature_description)
    anomaly_id = example[0]['anomaly_ID']
    moisture_anomaly = example[0]['moisture_anomalous_sensor']
    temp_anomaly = example[0]['temp_anomalous_sensor']
    battv_anomaly = example[0]['battv_anomalous_sensor']
    sensor_ind = tf.squeeze(tf.where(example[0]['sensor_ids'] == anomaly_id))
        
    if normalization == 'standarization':
        moisture_anomaly = (moisture_anomaly - tf.gather(example[0]['moisture_mean'], sensor_ind))/tf.gather(
            example[0]['moisture_std'], sensor_ind)
        temp_anomaly = (temp_anomaly - tf.gather(example[0]['temp_mean'], sensor_ind))/tf.gather(
            example[0]['temp_std'], sensor_ind)
        battv_anomaly = (battv_anomaly - tf.gather(example[0]['battv_mean'], sensor_ind))/tf.gather(
            example[0]['battv_std'], sensor_ind)
    elif normalization == 'scale_range':                                                              
        moisture_anomaly = moisture_anomaly/60
        temp_min = -20
        temp_max = 40
        temp_anomaly = (temp_anomaly - temp_min)/(temp_max - temp_min)
        battv_min = 2800
        battv_max = 3600
        battv_anomaly = (battv_anomaly - battv_min)/(battv_max - battv_min)
    elif normalization == 'scale':                                                              
        moisture_anomaly = (moisture_anomaly - tf.gather(example[0]['moisture_min'], sensor_ind))/(
            tf.gather(example[0]['moisture_max'], sensor_ind) - tf.gather(example[0]['moisture_min'], sensor_ind))
        temp_anomaly = (temp_anomaly - tf.gather(example[0]['temp_min'], sensor_ind))/(
            tf.gather(example[0]['temp_max'], sensor_ind) - tf.gather(example[0]['temp_min'], sensor_ind))
        battv_anomaly = (battv_anomaly - tf.gather(example[0]['battv_min'], sensor_ind))/(
            tf.gather(example[0]['battv_max'], sensor_ind) - tf.gather(example[0]['battv_min'], sensor_ind))
    elif normalization == 'median':
        moisture_anomaly = (moisture_anomaly - tf.gather(example[0]['moisture_median'], sensor_ind))/tf.gather(
            example[0]['moisture_median'], sensor_ind)
        temp_anomaly = (temp_anomaly - tf.gather(example[0]['temp_median'], sensor_ind))/tf.gather(
            example[0]['temp_median'], sensor_ind)
        battv_anomaly = (battv_anomaly - tf.gather(example[0]['battv_median'], sensor_ind))/tf.gather(
            example[0]['battv_median'], sensor_ind)
    elif normalization == 'rolling_median':
        moisture_anomaly = (moisture_anomaly - tf.gather(example[0]['moisture_rolling_median'], sensor_ind))
        temp_anomaly = (temp_anomaly - tf.gather(example[0]['temp_rolling_median'], sensor_ind))
        battv_anomaly = (battv_anomaly - tf.gather(example[0]['battv_rolling_median'], sensor_ind))
    elif normalization == 'rolling_median_fractional':
        moisture_anomaly = (moisture_anomaly - tf.gather(example[0]['moisture_rolling_median'], sensor_ind))/tf.gather(
            example[0]['moisture_rolling_median'], sensor_ind)
        temp_anomaly = (temp_anomaly - tf.gather(example[0]['temp_rolling_median'], sensor_ind))/tf.gather(
            example[0]['temp_rolling_median'], sensor_ind)
        battv_anomaly = (battv_anomaly - tf.gather(example[0]['battv_rolling_median'], sensor_ind))/tf.gather(
            example[0]['battv_rolling_median'], sensor_ind)
    elif normalization == 'rolling_mean':
        moisture_anomaly = (moisture - tf.gather(example[0]['moisture_rolling_mean'], sensor_ind))/tf.gather(
            example[0]['moisture_rolling_std'], sensor_ind)
        temp_anomaly = (temp_anomaly - tf.gather(example[0]['temp_rolling_mean'], sensor_ind))/tf.gather(
            example[0]['temp_rolling_std'], sensor_ind)
        battv_anomaly = (battv_anomaly - tf.gather(example[0]['battv_rolling_mean'], sensor_ind))/tf.gather(
            example[0]['battv_rolling_std'], sensor_ind)
    
    return moisture_anomaly, temp_anomaly, battv_anomaly, anomaly_id, example[0]['dates'], example[0]['anomaly_flag']


@tf.function
def prepare_batch_cml(batch_trsl1, batch_trsl2, batch_trsl1_anomalous_cml, batch_trsl2_anomalous_cml,\
    batch_cml_lat_a, batch_cml_lat_b, batch_cml_lon_a, batch_cml_lon_b, batch_distances,\
    batch_node_numb, batch_link_numb, batch_nodes, batch_neighbors, cml_ind, batch_anomaly_cml_id, batch_dates, anomaly_flags):
    
    cml_lat_a = batch_cml_lat_a.merge_dims(outer_axis=0, inner_axis=-1)
    cml_lat_b = batch_cml_lat_b.merge_dims(outer_axis=0, inner_axis=-1)
    cml_lon_a = batch_cml_lon_a.merge_dims(outer_axis=0, inner_axis=-1)
    cml_lon_b = batch_cml_lon_b.merge_dims(outer_axis=0, inner_axis=-1)
    distances = batch_distances.merge_dims(outer_axis=0, inner_axis=-1)
    trsl = tf.stack([batch_trsl1.merge_dims(outer_axis=0, inner_axis=-1),
                     batch_trsl2.merge_dims(outer_axis=0, inner_axis=-1)], 1)
    trsl_anomalous_cml = tf.stack([batch_trsl1_anomalous_cml, batch_trsl2_anomalous_cml], -1)
    num_nodes_per_timestep = batch_trsl1.row_lengths(2).merge_dims(outer_axis=0, inner_axis=-1)
    num_timesteps_per_sample = batch_trsl1.row_lengths(1)
    total_nodes_numb = tf.reduce_sum(num_nodes_per_timestep)
    link_numb = tf.repeat(batch_nodes.row_lengths(1), num_timesteps_per_sample)
    graph_indices = tf.range(tf.reduce_sum(num_timesteps_per_sample))
    sample_indices = tf.range(tf.shape(num_timesteps_per_sample)[0])
    sample_indicator = tf.repeat(sample_indices, tf.reduce_sum(batch_trsl1.row_lengths(2), 1))
    anomalous_cml_indicator = tf.repeat(cml_ind, batch_trsl1.row_lengths(1))    
    gather_indices = tf.repeat(graph_indices[:-1], link_numb[1:])
    increment = tf.cumsum(num_nodes_per_timestep[:-1])
    anomalous_cml_indicator = anomalous_cml_indicator + tf.cast(tf.pad(increment, [(1, 0)]), tf.int64)
    increment = tf.cast(tf.pad(tf.gather(increment, gather_indices), [(link_numb[0], 0)]), tf.int64)
    pair_indices = tf.stack([tf.tile(batch_nodes.merge_dims(outer_axis=1, inner_axis=2),
                                     [1, num_timesteps_per_sample[0]]).merge_dims(outer_axis=0, inner_axis=-1),
                             tf.tile(batch_neighbors.merge_dims(outer_axis=1, inner_axis=2),
                                     [1, num_timesteps_per_sample[0]]).merge_dims(outer_axis=0, inner_axis=-1)], 1)
    pair_indices = pair_indices + increment[:, tf.newaxis]
    joint_adjacency_matrix = tf.sparse.SparseTensor(pair_indices, tf.cast(tf.repeat(1, tf.shape(pair_indices)[0]), tf.float32),
                                   dense_shape=[total_nodes_numb, total_nodes_numb])
    trsl.set_shape([None, 2])
    trsl_anomalous_cml.set_shape([None, None, 2])
    anomaly_flags.set_shape([None])
    return trsl, trsl_anomalous_cml, cml_lat_a, cml_lat_b, cml_lon_a, cml_lon_b, joint_adjacency_matrix,\
    sample_indicator, distances, anomalous_cml_indicator, batch_anomaly_cml_id, batch_dates, anomaly_flags


@tf.function
def prepare_batch_soilnet(batch_moisture, batch_temp, batch_battv, batch_moisture_anomalous_sensor,\
                  batch_temp_anomalous_sensor, batch_battv_anomalous_sensor,\
    batch_lat, batch_lon, batch_distances, batch_depths,\
    batch_node_numb, batch_link_numb, batch_nodes, batch_neighbors, sensor_ind, batch_anomaly_sensor_id,\
                  batch_dates, anomaly_flags):
    
    lat = batch_lat.merge_dims(outer_axis=0, inner_axis=-1)
    lon = batch_lon.merge_dims(outer_axis=0, inner_axis=-1)
    distances = batch_distances.merge_dims(outer_axis=0, inner_axis=-1)
    depths = batch_depths.merge_dims(outer_axis=0, inner_axis=-1)
    features = tf.stack([batch_moisture.merge_dims(outer_axis=0, inner_axis=-1),
                         batch_temp.merge_dims(outer_axis=0, inner_axis=-1),
                     batch_battv.merge_dims(outer_axis=0, inner_axis=-1)], 1)
    features_anomalous_sensor = tf.stack([batch_moisture_anomalous_sensor, batch_temp_anomalous_sensor,
                                          batch_battv_anomalous_sensor], -1)
    num_nodes_per_timestep = batch_moisture.row_lengths(2).merge_dims(outer_axis=0, inner_axis=-1)
    num_timesteps_per_sample = batch_moisture.row_lengths(1)
    total_nodes_numb = tf.reduce_sum(num_nodes_per_timestep)
    link_numb = tf.repeat(batch_nodes.row_lengths(1), num_timesteps_per_sample)
    graph_indices = tf.range(tf.reduce_sum(num_timesteps_per_sample))
    sample_indices = tf.range(len(num_timesteps_per_sample))
    sample_indicator = tf.repeat(sample_indices, tf.reduce_sum(batch_moisture.row_lengths(2), 1))
    anomalous_sensor_indicator = tf.repeat(sensor_ind, batch_moisture.row_lengths(1))
    gather_indices = tf.repeat(graph_indices[:-1], link_numb[1:])
    increment = tf.cumsum(num_nodes_per_timestep[:-1])
    anomalous_sensor_indicator = anomalous_sensor_indicator + tf.cast(tf.pad(increment, [(1, 0)]), tf.int64)

    increment = tf.cast(tf.pad(tf.gather(increment, gather_indices), [(link_numb[0], 0)]), tf.int64)
    pair_indices = tf.stack([tf.tile(batch_nodes.merge_dims(outer_axis=1, inner_axis=2),
                                     [1, num_timesteps_per_sample[0]]).merge_dims(outer_axis=0, inner_axis=-1),
                             tf.tile(batch_neighbors.merge_dims(outer_axis=1, inner_axis=2),
                                     [1, num_timesteps_per_sample[0]]).merge_dims(outer_axis=0, inner_axis=-1)], 1)
    pair_indices = pair_indices + increment[:, tf.newaxis]
    joint_adjacency_matrix = tf.sparse.SparseTensor(pair_indices, tf.cast(tf.repeat(1, tf.shape(pair_indices)[0]), tf.float32),
                                   dense_shape=[total_nodes_numb, total_nodes_numb])
    features.set_shape([None, 3])
    features_anomalous_sensor.set_shape([None, None, 3])
    anomaly_flags.set_shape([None])
    return features, features_anomalous_sensor, lat, lon, joint_adjacency_matrix,\
    sample_indicator, distances, depths, anomalous_sensor_indicator, batch_anomaly_sensor_id,\
    batch_dates, anomaly_flags


def create_batched_dataset(dataset, preproc_config, shuffle=True, baseline=False):
    call_numb = tf.data.AUTOTUNE
    if preproc_config.ds_type == 'cml':
        if baseline == False:
            dataset = dataset.map(parse_tfrecord_fn)
            value = signature(parse_tfrecord_fn).parameters['normalization'].default
            batch_function = prepare_batch_cml
            wrapping_functions = [wrapper_batch_mapping, wrapper_batch_mapping_plot]
        else:
            dataset = dataset.map(parse_tfrecord_fn_baseline)
            value = signature(parse_tfrecord_fn_baseline).parameters['normalization'].default
            wrapping_functions = [wrapper_batch_mapping_baseline, wrapper_batch_mapping_baseline_plot]
    elif preproc_config.ds_type == 'soilnet':
        if baseline == False:
            dataset = dataset.map(parse_soilnet_tfrecord_fn)
            value = signature(parse_soilnet_tfrecord_fn).parameters['normalization'].default
            batch_function = prepare_batch_soilnet
            wrapping_functions = [wrapper_batch_mapping_soilnet, wrapper_batch_mapping_soilnet_plot]
        else:
            dataset = dataset.map(parse_soilnet_tfrecord_fn_baseline)
            value = signature(parse_soilnet_tfrecord_fn_baseline).parameters['normalization'].default
            wrapping_functions = [wrapper_batch_mapping_soilnet_baseline,
                                  wrapper_batch_mapping_soilnet_baseline_plot]
    
    if shuffle:
        dataset = dataset.shuffle(preproc_config.shuffle_size, seed=preproc_config.random_state, reshuffle_each_iteration=True)
    batched_dataset = dataset.batch(preproc_config.batch_size)
    if baseline == False:
        batched_dataset = batched_dataset.map(batch_function)
    preproc_config.normalization = value
    return batched_dataset, preproc_config, wrapping_functions


@tf.function
def wrapper_batch_mapping(trsl_out, trsl_anomalous_cml, cml_lat_a, cml_lat_b, cml_lon_a, cml_lon_b,
                          joint_adjacency_matrix, sample_indices, distances, cml_ind, batch_anomaly_cml_id, batch_dates,
                           anomaly_flags):
    return (trsl_out, trsl_anomalous_cml, cml_lat_a, cml_lat_b, cml_lon_a, cml_lon_b,
            joint_adjacency_matrix, sample_indices, cml_ind), (anomaly_flags)  


@tf.function
def wrapper_batch_mapping_plot(trsl_out, trsl_anomalous_cml, cml_lat_a, cml_lat_b, cml_lon_a, cml_lon_b,
                          joint_adjacency_matrix, sample_indices, distances, cml_ind, batch_anomaly_cml_id, batch_dates,
                           anomaly_flags):
    return (trsl_out, trsl_anomalous_cml, cml_lat_a, cml_lat_b, cml_lon_a, cml_lon_b, joint_adjacency_matrix,
            sample_indices, cml_ind, distances, batch_anomaly_cml_id, batch_dates), (anomaly_flags)


@tf.function
def wrapper_batch_mapping_baseline(batch_trsl1_anomalous_cml, batch_trsl2_anomalous_cml, batch_anomaly_id, batch_dates, anomaly_flags):
    trsl_anomalous_cml = tf.stack([batch_trsl1_anomalous_cml, batch_trsl2_anomalous_cml], -1)
    trsl_anomalous_cml.set_shape([None, None, 2])
    anomaly_flags.set_shape([None])
    return (trsl_anomalous_cml), (anomaly_flags)


@tf.function
def wrapper_batch_mapping_baseline_plot(batch_trsl1_anomalous_cml, batch_trsl2_anomalous_cml, batch_anomaly_cml_id, batch_dates, anomaly_flags):
    trsl_anomalous_cml = tf.stack([batch_trsl1_anomalous_cml, batch_trsl2_anomalous_cml], -1)
    trsl_anomalous_cml.set_shape([None, None, 2])
    anomaly_flags.set_shape([None])
    return (trsl_anomalous_cml, batch_anomaly_cml_id, batch_dates), (anomaly_flags)


def wrapper_batch_mapping_soilnet(features, features_anomalous_sensor, lat, lon,
                          joint_adjacency_matrix, sample_indices, distances, depths,
                          sensor_ind, batch_anomaly_sensor_id, batch_dates,
                           anomaly_flags):
    return (features, features_anomalous_sensor, lat, lon,
            joint_adjacency_matrix, sample_indices, sensor_ind), (anomaly_flags)  


def wrapper_batch_mapping_soilnet_plot(features, features_anomalous_sensor, lat, lon,
                          joint_adjacency_matrix, sample_indices, distances, depths,
                          sensor_ind, batch_anomaly_sensor_id, batch_dates,
                           anomaly_flags):
    return (features, features_anomalous_sensor, lat, lon, joint_adjacency_matrix,
            sample_indices, sensor_ind, batch_anomaly_sensor_id, batch_dates), (anomaly_flags) 


def wrapper_batch_mapping_soilnet_baseline(moisture_anomaly, temp_anomaly, battv_anomaly, batch_anomaly_sensor_id, batch_dates, anomaly_flags):
    features_anomalous_sensor = tf.stack([moisture_anomaly, temp_anomaly, battv_anomaly], -1)
    features_anomalous_sensor.set_shape([None, None, 3])
    anomaly_flags.set_shape([None])
    return (features_anomalous_sensor), (anomaly_flags)  


def wrapper_batch_mapping_soilnet_baseline_plot(moisture_anomaly, temp_anomaly, battv_anomaly, batch_anomaly_sensor_id, batch_dates, anomaly_flags):
    features_anomalous_sensor = tf.stack([moisture_anomaly, temp_anomaly, battv_anomaly], -1)
    features_anomalous_sensor.set_shape([None, None, 3])
    anomaly_flags.set_shape([None])
    return (features_anomalous_sensor, batch_anomaly_sensor_id, batch_dates), (anomaly_flags)  
    
def load_dataset_CV(preproc_config, test_split=0):
    preproc_config.split = test_split
    timestep_before = preproc_config.timestep_before
    timestep_after = preproc_config.timestep_after
    split_numb = preproc_config.dataset.split_numb
    file_list_original = glob.glob(os.path.join(preproc_config.dataset.tfrecords_dataset_dir,
                                             '{}_{}'.format(timestep_before, timestep_after), '**', '*.tfrec'), recursive=True)
    files_df = pd.DataFrame(file_list_original, columns =['file_paths'])
    
    files_df["filename"] = files_df['file_paths'].str.rsplit('/', n=1, expand=True)[1]
    files_df["file_numb"] = files_df["filename"].str.rsplit('_', n=1,
                                                            expand=True)[1].str.removesuffix(
        ".tfrec").astype('int64')
    if preproc_config.ds_type == 'cml':
        files_df["sensors_id"] = files_df["filename"].str.slice(stop=17)
    elif preproc_config.ds_type == 'soilnet':
        files_df["sensors_id"] = files_df["filename"].str.split('_', n=1, expand=True)[0].astype('int64')   
    
    total_files_numb = len(files_df.index)
    unique_sensors = files_df["sensors_id"].unique()
    sensors_numb = len(unique_sensors)
    max_file_numb = files_df["file_numb"].max()
    min_file_numb = files_df["file_numb"].min()
    
    split_indices = np.array_split(np.arange(min_file_numb, max_file_numb + 1), split_numb)
    test_indices = split_indices[test_split]
    train_indices = np.setdiff1d(np.concatenate(split_indices), test_indices)
    file_list_train = files_df.loc[files_df["file_numb"].isin(train_indices)]['file_paths'].tolist() 
    file_list_test = files_df.loc[files_df["file_numb"].isin(test_indices)]['file_paths'].tolist()
    random.shuffle(file_list_train)
    train_dataset = tf.data.TFRecordDataset(file_list_train)
    test_dataset = tf.data.TFRecordDataset(file_list_test)
    return train_dataset, test_dataset, preproc_config


def load_dataset(preproc_config, filter_func=False, **kwargs):
    """
    Load the dataset for training, validation, and testing.

    Args:
        preproc_config: The preprocessing configuration.
        filter_func: A function used to filter the dataset (optional).
        **kwargs: Additional keyword arguments to be passed to the filter function.

    Returns:
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        test_dataset: The testing dataset.
    """
    timestep_before = preproc_config.timestep_before
    timestep_after = preproc_config.timestep_after
    file_list_original = glob.glob(os.path.join(preproc_config.dataset.tfrecords_dataset_dir,
                                             '{}_{}'.format(timestep_before, timestep_after), '**', '*.tfrec'), recursive=True)
    train_frac = preproc_config.dataset.train_fraction
    val_frac = preproc_config.dataset.val_fraction
    files_df = pd.DataFrame(file_list_original, columns =['file_paths'])
    if len(files_df.index) == 0:
        raise ValueError('No files found in the directory: {}'.format(preproc_config.dataset.tfrecords_dataset_dir))
    files_df["filename"] = files_df['file_paths'].str.rsplit('/', n=1, expand=True)[1]
    files_df["file_numb"] = files_df["filename"].str.rsplit('_', n=1,
                                                            expand=True)[1].str.removesuffix(
        ".tfrec").astype('int64')
    
    if preproc_config.ds_type == 'cml':
        files_df["sensors_id"] = files_df["filename"].str.slice(stop=17)
    elif preproc_config.ds_type == 'soilnet':
        files_df["sensors_id"] = files_df["filename"].str.split('_', n=1, expand=True)[0].astype('int64')   
    total_files_numb = len(files_df.index)
    # Split the file list based on the dates
    unique_sensors = files_df["sensors_id"].unique()
    sensors_numb = len(unique_sensors)
    min_sensor_numb = files_df["file_numb"].min()
    # get the length of time series, which will be used for training
    train_timeseries_length = int(min_sensor_numb + np.round(total_files_numb/sensors_numb*train_frac))
    val_timeseries_length = int(total_files_numb/sensors_numb*val_frac)
    file_list_train = files_df.loc[files_df["file_numb"]<train_timeseries_length]['file_paths'].tolist() 
    file_list_val = files_df.loc[(files_df["file_numb"]>=train_timeseries_length) & 
    (files_df["file_numb"]<train_timeseries_length + val_timeseries_length)]['file_paths'].tolist()
    file_list_test = files_df.loc[(files_df["file_numb"]>=train_timeseries_length + val_timeseries_length)]['file_paths'].tolist()
    random.shuffle(file_list_train)
    print('Number of training files: {}'.format(len(file_list_train)))
    print('Number of validation files: {}'.format(len(file_list_val)))
    print('Number of test files: {}'.format(len(file_list_test)))
    random.shuffle(file_list_val)
    
    # apply filter if desired
    if filter_func:
        file_list_test = filter_func(file_list_test, **kwargs)
        file_list_train = filter_func(file_list_train, **kwargs)
        file_list_val = filter_func(file_list_val, **kwargs)
        print('Number of training files after filtering: {}'.format(len(file_list_train)))
        print('Number of validation files after filtering: {}'.format(len(file_list_val)))
        print('Number of test files after filtering: {}'.format(len(file_list_test)))

    train_dataset = tf.data.TFRecordDataset(file_list_train)
    val_dataset = tf.data.TFRecordDataset(file_list_val)
    test_dataset = tf.data.TFRecordDataset(file_list_test)
    return train_dataset, val_dataset, test_dataset

    
def float_feature_from_list(value_list):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def bytes_feature_from_list(value_list):
    return tf.train.Feature(bytes_list=tf.train.BytesList(
                            value=[m.encode('utf-8') for m in value_list]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_feature_from_list(value_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


def float_featurelist_from_list(value_list):
    return tf.train.FeatureList(feature=[tf.train.Feature(
                                float_list=tf.train.FloatList(value=x))
                                          for x in value_list])


def float_featurelist(value_list):
    return tf.train.FeatureList(feature=[tf.train.Feature(
                                float_list=tf.train.FloatList(value=[x]))
                                          for x in value_list])


def int64_featurelist(value_list):
    return tf.train.FeatureList(feature=[tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[x]))
                                          for x in value_list])


def coordinates_featurelist(value_list, sequence_length):
    return tf.train.FeatureList(feature=[tf.train.Feature(
      float_list=tf.train.FloatList(value=value_list))
                                          for x in range(sequence_length)])


def create_sensors_ncfiles(ds, preproc_config):
    random.seed(preproc_config.random_state)
    max_dist = preproc_config.graph.max_sample_distance
    max_depth = preproc_config.graph.max_depth
    # Create output dir if doesn't exist
    if not os.path.exists(preproc_config.dataset.ncfiles_dir):
        os.makedirs(preproc_config.dataset.ncfiles_dir)     
    if preproc_config.ds_type == 'cml':
        # Replace outstanding high values with nan
        ds['TL_2'] = ds.TL_2.where(ds.TL_2.values<200, other=np.nan)
        ds['TL_1'] = ds.TL_1.where(ds.TL_1.values<200, other=np.nan)
        print('High (over 200 dB) values replaced with NaN')
        features_list = ['TL_1', 'TL_2']
        flag_vars = ['Jump', 'Dew', 'Fluctuation', 'Unknown anomaly']
        # get flagged CMLs ID
        flagged_sensors = ds.where(ds.flagged, drop=True).sensor_id.values
        interpolation_max_time = '5min'
        ds_type = 'cml'
    elif preproc_config.ds_type == 'soilnet':
        ds = ds.dropna(dim='sensor_id', subset=['latitude', 'longitude'])
        flagged_sensors = select_sensors(ds)
        features_list = ['moisture', 'temp', 'battv']
        interpolation_max_time = '60min'
        flag_vars=None
        ds_type = 'soilnet'
    # interpolate data if desired
    if preproc_config.dataset.interpolate:
        ds = interpolate_features(ds, features=features_list, 
                                      interpolation_max=interpolation_max_time, method="linear")
    # Create target
    ds['target'] = ('sensor_id', 'time'), create_target(ds, ds_type=preproc_config.ds_type, flag_vars=flag_vars)
    # Calculate distances between sensors
    distances_matrix = compute_distance_matrix(ds, ds_type)
    print('Distance matrix calculated')
    if preproc_config.ds_type == 'cml':
        depths_matrix = None
    elif preproc_config.ds_type == 'soilnet':
        depths_matrix = compute_depth_matrix(ds)
        print('Depth matrix calculated')
        flag_vars = []
        for k in ds.keys():
            if 'flag' in k:
                flag_vars.append(k)
    ds = ds.drop(flag_vars)
    for sensor in flagged_sensors:
        print('Processing sensor: {}'.format(sensor))
        neighbors = get_neighbors(distances_matrix, sensor, max_dist, max_depth, preproc_config.ds_type, depths= depths_matrix)
        # optional reduction of distance matrix for easy creation of adjacency matrix
        reduced_distances_matrix = distances_matrix.sel({'sensor_id': neighbors}) 
        
        reduced_ds = ds.sel({'sensor_id': neighbors})
        # target is reduced to be only from the sensor of interest
        reduced_ds['target'] = 'time', reduced_ds.target.sel({'sensor_id': sensor}).values
        # flagged now indicates which sensor is related to the target (True) and which are neighbors (False)
        reduced_ds['flagged'] = 'sensor_id', reduced_ds.sensor_id.values==sensor 
        reduced_ds = reduced_ds.assign(distances=reduced_distances_matrix)
        if preproc_config.ds_type == 'cml':
            path_out = os.path.join(preproc_config.dataset.ncfiles_dir, '%s.nc' % sensor)
        elif preproc_config.ds_type == 'soilnet':
            reduced_depths_matrix = depths_matrix.sel({'sensor_id': neighbors})
            reduced_ds = reduced_ds.assign(depths=reduced_depths_matrix)
            box = reduced_ds.where(reduced_ds.flagged, drop=True).box_id.values[0]
            path_out = os.path.join(preproc_config.dataset.ncfiles_dir, '{}_{}.nc'.format(box, sensor))
        
        reduced_ds.attrs['anomalous_sensor_id'] = sensor
        reduced_ds.to_netcdf(path_out)


def select_sensors(ds):
    ds_box = ds.groupby(ds.box_id)
    sensors_list = []
    for box, gb in ds_box:
        lp = gb.moisture.notnull().sum(dim='time')
        sensors_list.append(random.choice(lp.where(lp==lp.max(), drop=True).sensor_id.values))
    return sensors_list


def create_target(ds, flag_vars=None, min_experts=3, ds_type='clm', flags_type='manual'):
    if ds_type == 'cml':
        ds_new = ds.copy()
        features = []
        for feature in flag_vars:
            features.append(ds_new[feature].astype(int).sum(dim='expert').values>=min_experts)
        target = np.any(np.stack(features, axis=0), axis=0)
    elif ds_type == 'soilnet':
        if flags_type == 'manual':
            target = xr.where(ds.moisture_flag_OK==True, 0, np.nan).values
            target[ds.moisture_flag_Manual==True] = 1
        else:
            # for all flags
            target = ds.moisture_flag_no_label.where(ds.moisture_flag_no_label==False).values
            target[(ds.moisture_flag_no_label==False) & (ds.moisture_flag_OK==False)] = True
    return target


def compute_distance_matrix(ds, ds_type='cml'):
    distances = np.zeros([len(ds.sensor_id), len(ds.sensor_id)])
    if ds_type == 'cml':
        lat = (ds.site_a_latitude.values + ds.site_b_latitude.values)/2
        lon = (ds.site_a_longitude.values + ds.site_b_longitude.values)/2
    elif ds_type == 'soilnet':
        lat = ds.latitude.values
        lon = ds.longitude.values
    
    for i in range(len(ds.sensor_id)):
        coords_1 = (lat[i], lon[i])
        for j in range(i + 1, len(lat)):
            coords_2 = (lat[j], lon[j])
            # Calculate distances in meters
            dist = geopy.distance.geodesic(coords_1, coords_2).km*1000
            distances[i, j] = dist
            distances[j, i] = dist
            
    da = xr.DataArray(distances, dims=['sensor_id', 'sensor_id'])
    da = da.assign_coords({'sensor_id':('sensor_id', ds.sensor_id.values),})
    return da


def get_neighbors(distances, sensor_id, max_dist=60, max_depth=0.3, ds_type='cml', depths=None):
    if ds_type == 'cml':
        neighbors = distances.sensor_id.values[distances.values[np.where(distances.sensor_id==sensor_id)[0][0]]<=max_dist]
    elif ds_type == 'soilnet':
        neighbors = distances.sensor_id.values[((distances.values[np.where(distances.sensor_id==sensor_id)[0][0]]<=max_dist) &
        (depths.values[np.where(depths.sensor_id==sensor_id)[0][0]]==0)) |
        ((distances.values[np.where(distances.sensor_id==sensor_id)[0][0]]==0) &
       (depths.values[np.where(depths.sensor_id==sensor_id)[0][0]]<=max_depth))]
    return list(neighbors)                             


def compute_depth_matrix(ds):
    depth = ds.depth.values
    # calculate depth differences and establish links based on it
    xv, yv = np.meshgrid(depth, depth)
    depth_matrix = np.abs(xv - yv)
    #return depth_matrix
    da = xr.DataArray(depth_matrix, dims=['sensor_id', 'sensor_id'])
    da = da.assign_coords({'sensor_id':('sensor_id', ds.sensor_id.values),})
    return da


def interpolate_features(ds, features, interpolation_max, method="linear"):
    """
    Scipy interpolate up to a gap size of 'interpolation_max' timesteps.
    """
    ds_new = ds.copy()
    for feature in features:
        feature_x = ds_new[feature].where(~np.isnan(ds_new[feature]), other=np.nan)
        feature_x = feature_x.interpolate_na(dim="time", max_gap=interpolation_max, method=method)
        ds_new[feature] = feature_x
    return ds_new