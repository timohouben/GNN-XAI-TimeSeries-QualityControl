The code in this directory has been used to produce the XAI results in the following paper:

 **[Interpretable Quality Control of Sparsely Distributed Environmental Sensor Networks Using Graph Neural Networks](https://journals.ametsoc.org/view/journals/aies/4/1/AIES-D-24-0032.1.xml?tab_body=supplementary-materials)** <br>
ELŻBIETA LASOTA, TIMO HOUBEN, JULIUS POLZ, LENNART SCHMIDT, LUCA GLAWION, DAVID SCHÄFER, JAN BUMBERGER, AND CHRISTIAN CHWALA <br>
published in [Artificial Intelligence for the Earth Systems](https://journals.ametsoc.org/view/journals/aies/aies-overview.xml) in January 2025. <br>

The analysis was done on March 18th 2024 so corresponding scripts contain this date:

`/notebooks/run_integrated_gradients_20240318.py`
`/notebooks/run_integrated_gradients_analyser_20240318.py`
`/config/model_config_20240318.yml`
`/config/preprocessing_config_20240318.yml`
`/config/xai_config_20240318.yml`


The code of the GNN model construction deviates from the code in the upper level of this repository, since the model was slightly adapded but the XAI part remained at a previous state, thus the above mentioned XAI scripts are not compatible with the GNN model in the upper level of this repository.