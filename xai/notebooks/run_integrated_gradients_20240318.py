import sys
import os

parent_dir = os.path.abspath("..")
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import libs.integrated_gradients as ig

preproc_config_path = "../libs/config/preprocessing_config_20240318.yml"
model_config_path = "../libs/config/model_config_20240318.yml"
xai_config_path = "../libs/config/xai_config_20240318.yml"

ig_xplain = ig.IntegratedGradientsExplainer(
    preproc_config_path, model_config_path, xai_config_path
)


ig_xplain.prepare_data()
ig_xplain.get_gradients()

sensors = ['BY4168_2_BY4036_4']
time_from = '2019-07-16 05:00'               
time_to = '2019-07-20 14:00'                 
ig_xplain.plot_ig_heatmap_from_directory(sensors=sensors, time_from=time_from, time_to=time_to)

sensors = ['DO1695_2_DO6017_2']
time_from = '2019-07-26 15:00'               
time_to = '2019-07-31 00:00'    
ig_xplain.plot_ig_heatmap_from_directory(sensors=sensors, time_from=time_from, time_to=time_to)

print("Done")