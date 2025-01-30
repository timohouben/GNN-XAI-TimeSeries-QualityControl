import sys
import os
import numpy as np
import matplotlib.pyplot as plt

parent_dir = os.path.abspath("..")
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import libs.integrated_gradients_analyser as iga

preproc_config_path = "../libs/config/preprocessing_config_20240318.yml"
model_config_path = "../libs/config/model_config_20240318.yml"
xai_config_path = "../libs/config/xai_config_20240318.yml"
ig_analyser = iga.IntegrateGradientsAnalyser(
    preproc_config_path, model_config_path, xai_config_path
)


ig_analyser.get_overview()

ig_analyser.spatial_aggregate_gradients()
ig_analyser.plot_spatial_aggregated_gradients()

sensor = 'BY4168_2_BY4036_4'
time_from = '2019-07-16 05:00'               
time_to = '2019-07-20 14:00'                 
ig_analyser.create_videos(sensor=sensor, time_from=time_from, time_to=time_to)
ig_analyser.plot_agg_samples_over_time(sensor=sensor, time_from= time_from, time_to=time_to, agg_type='mean', norm_by_prediction=False, cbar_limits=(-0.015, 0.015))
ig_analyser.plot_agg_samples_over_time(sensor=sensor, time_from= time_from, time_to=time_to, agg_type='mean', norm_by_prediction=True, cbar_limits=(-0.015, 0.015))

sensor = 'DO1695_2_DO6017_2'
time_from = '2019-07-26 15:00'               
time_to = '2019-07-31 00:00'
ig_analyser.create_videos(sensor=sensor, time_from=time_from, time_to=time_to)
ig_analyser.plot_agg_samples_over_time(sensor=sensor, time_from= time_from, time_to=time_to, agg_type='mean', norm_by_prediction=False, cbar_limits=(-0.002, 0.002))
ig_analyser.plot_agg_samples_over_time(sensor=sensor, time_from= time_from, time_to=time_to, agg_type='mean', norm_by_prediction=True, cbar_limits=(-0.005, 0.005))