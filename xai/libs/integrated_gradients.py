import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import SymLogNorm

import numpy as np
import itertools
import sys
import tensorflow as tf
import xarray as xr
import omegaconf as oc
import pandas as pd
import shutil
from pathlib import Path
import os
import glob


sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

from libs.metrics import matthews_correlation
from libs.visualize import timeseries_figure
from libs.preprocessing_functions import (
    load_dataset,
    create_batched_dataset,
    create_sensors_ncfiles,
    create_tfrecords_dataset,
    wrapper_batch_mapping,
    wrapper_batch_mapping_plot,
)

tf.config.run_functions_eagerly(False)


# prefix explanation
# fn_ = filename
# fp_ = filepath
# fns_ = list of filenames
# fps_ = list of filepaths
# dn_ = directory name
# dp_ = directory path
# dns_ = list of directory names
# dps_ = list of directory paths


DATA_INDEX_BATCH = {
    "features": (0, 0),
    "anomalous_ts": (0, 1),
    "lat_a": (0, 2),
    "lat_b": (0, 3),
    "lon_a": (0, 4),
    "lon_b": (0, 5),
    "adjacency_matrix": (0, 6),
    "sample_indices": (0, 7),
    "sensor_ind": (0, 8),
    "anomaly_flags": 1,
}

DATA_INDEX_BATCH_PLOT = {
    "features": (0, 0),
    "anomalous_ts": (0, 1),
    "lat_a": (0, 2),
    "lat_b": (0, 3),
    "lon_a": (0, 4),
    "lon_b": (0, 5),
    "adjacency_matrix": (0, 6),
    "sample_indices": (0, 7),
    "sensor_ind": (0, 8),
    "distances": (0, 9),
    "batch_anomaly_sensor_id": (0, 10),
    "batch_dates": (0, 11),
    "anomaly_flags": 1,
}

COLOR_MAP = {
    (0, 0): "blue",  # True Negative, TN
    (0, 1): "orange",  # False Positive, FP
    (1, 0): "red",  # False Negative, FN
    (1, 1): "green",  # True Positive, TP
}

COLOR_NAMES = {
    (0, 0): "True Negative",
    (0, 1): "False Positive",
    (1, 0): "False Negative",
    (1, 1): "True Positive",
}


class IntegratedGradientsExplainer:
    """

    TO DO : IMPROVE DESCRIPTION

    Integrated Gradients Analysis Class

    This class is designed to perform the Integrated Gradients analysis on a given dataset,
    using the provided configuration files for preprocessing, model, and XAI.
    It creates the necessary output directories and prepares the data for analysis.

    The gradients are calculated for a full batch of data and then unwrapped for
    each sample in the batch.

    Args:
        preprocessing_config_path (str): Path to the preprocessing configuration file.
        model_config_path (str): Path to the model configuration file.
        xai_config_path (str): Path to the XAI configuration file.
    """

    ############################################################################
    # INITIALISATION
    ############################################################################
    def __init__(self, fp_preproc_config, fp_model_config, fp_xai_config):
        # Load configuration files
        # ----------------------------------------------------------------------
        self.fp_preproc_config = fp_preproc_config
        self.fp_model_config = fp_model_config
        self.fp_xai_config = fp_xai_config
        self.preproc_config = oc.OmegaConf.load(self.fp_preproc_config)
        self.model_config = oc.OmegaConf.load(self.fp_model_config)
        self.xai_config = oc.OmegaConf.load(self.fp_xai_config)

        # model and data
        # ----------------------------------------------------------------------
        self.model = None
        self.test_dataset_batchwrapfetch = None
        self.train_dataset_batchwrapfetch = None
        self.val_dataset_batchwrapfetch = None

        # changing variables for each processed batch
        # ----------------------------------------------------------------------
        self.sample_indices = None
        self.baseline = None
        self.alphas = None
        self.one_batch = None
        # target information of one batch
        self.sensor_ids, self.anomaly_dates, self.anomaly_flags_true = None, None, None

        # sample indexes for which the analysis should be performed
        self.sample_indexes = None

        # ig results
        # ----------------------------------------------------------------------
        # gradients for each interpolation step
        # contains gradients for features and anomalous time series
        self.path_predictions = None
        self.path_gradients_features = None
        self.path_gradients_anom_ts = None

        # unwrapped data and gradients
        # ----------------------------------------------------------------------
        # features, first element of the model input dict
        self.features_unwrapped = None

        # variables form the batch for plotting
        # ----------------------------------------------------------------------
        self.features_plot = None
        self.test_dataset_batchwrapfetchplot = None
        self.train_dataset_batchwrapfetchplot = None
        self.val_dataset_batchwrapfetchplot = None

        # changing variables for each sample iteration
        # ----------------------------------------------------------------------
        self._current_output_dir = None
        self._current_file_name = None

        # create output directory
        # ----------------------------------------------------------------------
        self.output_dir = self._create_output_dir(
            self.xai_config.output_dir,
            self.xai_config.project,
            self.preproc_config.ds_type,
            self.xai_config.integrated_gradients.dataset,
            "",
        )

        # run as array job if variable is available
        # ----------------------------------------------------------------------
        try:
            # workerid for array job
            self.workerid = int(os.environ["SLURM_ARRAY_TASK_ID"])
            # slurm task id
            self.n_worker = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        except KeyError:
            self.workerid = None
            self.n_worker = None

        # copy the configuration files to the output directory
        self._copy_config_files()

        # set random seed
        tf.random.set_seed(self.xai_config.integrated_gradients.random_seed)
        np.random.seed(self.xai_config.integrated_gradients.random_seed)

        # Load model
        # ----------------------------------------------------------------------
        print(f"! - Loading Model from {self.model_config.model_path}")
        self.model = tf.keras.models.load_model(
            self.model_config.model_path, compile=False
        )

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=self.model_config.optimizer,
            metrics=[
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.TruePositives(name="tp"),
                tf.keras.metrics.FalsePositives(name="fp"),
                tf.keras.metrics.TrueNegatives(name="tn"),
                tf.keras.metrics.FalseNegatives(name="fn"),
            ],
        )

    ############################################################################
    # METHODS FOR PATH HANDLING AND LOGGING
    ############################################################################
    def _create_output_dir(
        self,
        output_dir,
        project,
        ds_type,
        dataset,
        current_sensor_id="",
        current_anomaly_date="",
        true="",
        pred="",
    ):
        """
        Created directory structure based on parameters.
        to this directory.

        output_dir: str
            User defined output directory for saving the results.
        project: str
            Name of the project to be specified by the user.
        ds_type: str
            Type of the dataset.
        dataset: str
            Data to be used. Either train, test or val dataset.
        current_sensor_id: str
            Sensor ID of the sensor (sample) for which the analysis is performed.
        current_anomaly_date: str
            Date of the anomaly (sample) for which the analysis is performed.
        true: str
            True label of the anomaly.
        pred: str
            Predicted label of the anomaly.
        """
        if isinstance(current_anomaly_date, pd.Timestamp):
            current_anomaly_date = current_anomaly_date.strftime("%Y%m%d_%H%M%S")

        if current_sensor_id == "" and current_anomaly_date == "":
            output_dir = self._get_save_path(output_dir, project, ds_type, dataset)
        else:
            output_dir = self._get_save_path(
                output_dir,
                project,
                ds_type,
                dataset,
                current_sensor_id,
                current_anomaly_date,
                true,
                pred,
            )
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        return output_dir

    def _get_save_path(
        self,
        output_dir,
        project,
        ds_type,
        dataset,
        current_sensor_id="",
        current_anomaly_date="",
        true="",
        pred="",
    ):
        """
        Get the save path for the integrated gradients.

        Args:
        output_dir: str
            The output directory.
        project: str
            The project name.
        ds_type: str
            The type of the dataset.
        dataset: str
            The dataset name.
        current_sensor_id: str, optional
            The current sensor ID. Defaults to "".
        current_anomaly_date: str or pd.Timestamp, optional
            The current anomaly date. Defaults to "".
        true: str, optional
            The true value. Defaults to "".
        pred: str, optional
            The predicted value. Defaults to "".

        Returns:
            Path: The save path for the integrated gradients.
        """
        if isinstance(current_anomaly_date, pd.Timestamp):
            current_anomaly_date = current_anomaly_date.strftime("%Y%m%d_%H%M%S")
        if current_sensor_id == "" and current_anomaly_date == "":
            save_path = Path(
                output_dir, "integrated_gradients", project, ds_type, dataset
            )
        else:
            save_path = Path(
                output_dir,
                "integrated_gradients",
                project,
                ds_type,
                dataset,
                current_sensor_id,
                current_sensor_id
                + "_"
                + current_anomaly_date
                + "_"
                + str(true)
                + "_"
                + str(pred),
            )
        return save_path

    def _get_file_name(
        self,
        project,
        ds_type,
        dataset,
        current_sensor_id="",
        current_anomaly_date="",
        true="",
        pred="",
    ):
        """
        Get the file name based on the project, dataset, current sensor ID, current anomaly date, true value, and predicted value.

        Args:
        project: str
            The project name.
        ds_type: str
            The type of the dataset.
        dataset: str
            The dataset name.
        current_sensor_id: str, optional
            The current sensor ID. Defaults to "".
        current_anomaly_date: str or pd.Timestamp, optional
            The current anomaly date. Defaults to "".
        true: str, optional
            The true value. Defaults to "".
        pred: str, optional
            The predicted value. Defaults to "".

        Returns:
        str: The file name.
        """
        if isinstance(current_anomaly_date, pd.Timestamp):
            current_anomaly_date = current_anomaly_date.strftime("%Y%m%d_%H%M%S")
        if current_sensor_id == "" and current_anomaly_date == "":
            file_name = project + "_" + ds_type + "_" + dataset
        else:
            file_name = (
                project
                + "_"
                + ds_type
                + "_"
                + dataset
                + "_"
                + current_sensor_id
                + "_"
                + current_anomaly_date
                + "_"
                + str(true)
                + "_"
                + str(pred)
            )
        return file_name

    def log_file(
        self,
        output_dir,
        project,
        ds_type,
        dataset,
        batch_id,
        index,
        current_cml_id,
        current_anomaly_date,
    ):
        # concatinate the input parameters to a string and append to a file
        file_name = "log.txt"
        log_dir = Path(
            output_dir, "integrated_gradients", project, ds_type, dataset, "log"
        )
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        log_file = Path(log_dir, file_name)
        with open(log_file, "a") as f:
            f.write(
                f"{project},{ds_type},{dataset},{batch_id},{index},{current_cml_id},{current_anomaly_date}\n"
            )

    ############################################################################
    # OTHER HELPER METHODS
    ############################################################################
    def _get_nested_list_element(self, index, nested_list):
        """
        Retrieves an element from a nested list based on the given index.

        Args:
            index (int or tuple): The index of the element to retrieve. If the index is a tuple, it is assumed to be a nested index.
            nested_list (list): The nested list from which to retrieve the element.

        Returns:
            The element at the specified index in the nested list.

        """
        if isinstance(index, tuple):  # Check if index is a tuple for nested list
            result = nested_list[index[0]][index[1]]
        else:  # Handle non-nested index
            result = nested_list[index]
        return result

    @staticmethod
    def _split_work(file_list, workerid, n_worker):
        """
        Split the work for the workers of an array job.

        Args:
            workerid (int): The ID of the worker.
            file_list (list): The list of files to be split.
            workers (int): The total number of workers.

        Returns:
            list: The list of files assigned to the current worker.
        """
        file_list_worker = []
        for i, file in enumerate(file_list):
            if i % n_worker == workerid:
                file_list_worker.append(file)
        return file_list_worker

    def _print_shapes(self, data, prefix=""):
        """
        Prints the shapes of the given data recursively.

        Parameters
        ----------

        data: list or any
            The data to print the shapes of.
        prefix: str
            The prefix to add to the shape information.

        Returns
        -------

        None
        """
        if isinstance(data, list):
            for i, sub_data in enumerate(data):
                self._print_shapes(sub_data, prefix=f"{prefix}[{i}]")
        else:
            print(f"{prefix}: {len(data)}, Index: {prefix}")

    def _copy_config_files(self):
        # copy the config files to the output directory
        dst = Path(
            self.xai_config.output_dir, "integrated_gradients", self.xai_config.project
        )
        srcs = [self.fp_preproc_config, self.fp_model_config, self.fp_xai_config]
        for src in srcs:
            shutil.copy(src, Path(dst, Path(src).name))

    def _tuple_to_list(self, nested_tuple):
        return [
            self._tuple_to_list(item) if isinstance(item, (tuple, list)) else item
            for item in nested_tuple
        ]

    def _list_to_tuple(self, nested_list):
        return tuple(
            self._list_to_tuple(item) if isinstance(item, (list, tuple)) else item
            for item in nested_list
        )

    @staticmethod
    def _get_confusion_matrix_indexes(true_values, predicted_values, configurations):
        """
        Get indexes based on the requested configurations of the confusion matrix.

        Parameters:
        - true_values: List of true values (0 or 1)
        - predicted_values: List of predicted values (0 or 1)
        - configurations: List of requested configurations (e.g., ['TP', 'FP'])

        Returns:
        - List of indexes corresponding to the requested configurations
        """

        # Initialize an empty list to store indexes
        indexes = []

        # Iterate through the requested configurations and get the corresponding indexes
        for config in configurations:
            if config == "TP":
                indexes.extend(
                    [
                        i
                        for i, (t, p) in enumerate(zip(true_values, predicted_values))
                        if t == 1 and p == 1
                    ]
                )
            elif config == "TN":
                indexes.extend(
                    [
                        i
                        for i, (t, p) in enumerate(zip(true_values, predicted_values))
                        if t == 0 and p == 0
                    ]
                )
            elif config == "FP":
                indexes.extend(
                    [
                        i
                        for i, (t, p) in enumerate(zip(true_values, predicted_values))
                        if t == 0 and p == 1
                    ]
                )
            elif config == "FN":
                indexes.extend(
                    [
                        i
                        for i, (t, p) in enumerate(zip(true_values, predicted_values))
                        if t == 1 and p == 0
                    ]
                )

        return indexes

    def _get_current_target_info(self, sample_index):
        """
        Retrieves information about the current target for a given sample index.

        Parameters:
            sample_index (int): The index of the sample.

        Returns:
            tuple: A tuple containing the following information:
                - current_sensor_id (str): The current sensor ID.
                - current_anomaly_dates (numpy.ndarray): An array of anomaly dates.
                - current_anomaly_date (pandas.Timestamp): The current anomaly date.
                - current_anomaly_flag_true (bool): The flag indicating if the anomaly is true.
        """
        current_sensor_id = self.sensor_ids[sample_index]
        current_sensor_id = current_sensor_id.numpy().decode("utf-8")
        current_anomaly_dates = self.anomaly_dates[sample_index]
        current_anomaly_dates = np.vectorize(lambda x: x.decode("utf-8"))(
            current_anomaly_dates.numpy()
        )
        current_anomaly_dates = np.vectorize(
            lambda x: pd.to_datetime(x, format="%Y-%m-%dT%H:%M:%S")
        )(current_anomaly_dates)

        # TODO:
        # CHECK IF ig_xplain.preproc_config.timestep_before is always the correct index!!

        current_anomaly_date = current_anomaly_dates[
            self.preproc_config.timestep_before
        ]
        current_anomaly_flag_true = self.anomaly_flags_true[sample_index].numpy()

        return (
            current_sensor_id,
            current_anomaly_dates,
            current_anomaly_date,
            current_anomaly_flag_true,
        )

    ############################################################################
    # DATA LOADING/EXTRACTION/PREPARATION
    ############################################################################
    def prepare_data(self):
        """
        Prepare the data for training and evaluation.

        This method performs the following steps:
        1. Create NetCDF files if specified in the configuration.
        2. Create TFRecords files if specified in the configuration.
        3. Load TFRecords datasets.
        4. Create batches of data from TFRecords for predicting and plotting.
        5. Evaluate the model with the test dataset if specified in the configuration.

        Returns:
            None
        """
        # Create NetCDF files
        # ------------------------------------------------------------------------------
        # with flagged sensor and selected neighbours - separately for each sensor
        # Load raw data
        if self.preproc_config.dataset.create_nc_files:
            raw_ds = xr.open_dataset(
                self.preproc_config.dataset.raw_dataset_path
            ).load()
            # Create NetCDF files
            create_sensors_ncfiles(raw_ds, self.preproc_config)

        # Create TFRecords files
        # ------------------------------------------------------------------------------
        # separeatly for each sensor and split the dataset
        if self.preproc_config.dataset.create_tfrecords_dataset:
            create_tfrecords_dataset(self.preproc_config)

        # Load TFRecords datasets
        # ------------------------------------------------------------------------------
        print(
            "! - Loading TFRecords datasets from directory",
            self.preproc_config.dataset.tfrecords_dataset_dir,
        )

        if self.workerid is None:
            train_dataset, val_dataset, test_dataset = load_dataset(self.preproc_config)
        else:
            print("Distributing work based on number of files and workers.")
            print(f"! - Worker {self.workerid} of {self.n_worker} workers")
            train_dataset, val_dataset, test_dataset = load_dataset(
                self.preproc_config,
                filter_func=self._split_work,
                workerid=self.workerid,
                n_worker=self.n_worker,
            )

        # Set the number of parallel calls to tf.data.experimental.AUTOTUNE
        call_numb = tf.data.AUTOTUNE

        # Create batches of data from TFRrecords for predicting and plotting
        # ------------------------------------------------------------------------------
        print("! - Creating batches of data from TFRecords")
        if self.xai_config.integrated_gradients.dataset == "test":
            self.test_dataset_batchwrapfetch = (
                create_batched_dataset(
                    test_dataset, self.preproc_config, shuffle=False
                )[0]
                .map(wrapper_batch_mapping, call_numb)
                .prefetch(call_numb)
            )
            self.test_dataset_batchwrapfetchplot = (
                create_batched_dataset(
                    test_dataset, self.preproc_config, shuffle=False
                )[0]
                .map(wrapper_batch_mapping_plot, call_numb)
                .prefetch(call_numb)
            )
        elif self.xai_config.integrated_gradients.dataset == "val":
            self.val_dataset_batchwrapfetch = (
                create_batched_dataset(val_dataset, self.preproc_config, shuffle=False)[
                    0
                ]
                .map(wrapper_batch_mapping, call_numb)
                .prefetch(call_numb)
            )
            self.val_dataset_batchwrapfetchplot = (
                create_batched_dataset(val_dataset, self.preproc_config, shuffle=False)[
                    0
                ]
                .map(wrapper_batch_mapping_plot, call_numb)
                .prefetch(call_numb)
            )
        elif self.xai_config.integrated_gradients.dataset == "train":
            self.train_dataset_batchwrapfetch = (
                create_batched_dataset(
                    train_dataset, self.preproc_config, shuffle=False
                )[0]
                .map(wrapper_batch_mapping, call_numb)
                .prefetch(call_numb)
            )
            self.train_dataset_batchwrapfetchplot = (
                create_batched_dataset(
                    train_dataset, self.preproc_config, shuffle=False
                )[0]
                .map(wrapper_batch_mapping_plot, call_numb)
                .prefetch(call_numb)
            )

        # Evaluate Model with test dataset
        # ------------------------------------------------------------------------------
        if self.xai_config.integrated_gradients.evaluate_model == True:
            print("! - Evaluating Model")
            results = self.model.evaluate(self.test_dataset_batchwrapfetch)
            self.model.metrics_names
            print(
                "! - Model accuracy: {:.3},\nrecall: {:.3},\nprecision: {:.3},\n".format(
                    results[2], results[1], results[3]
                )
            )
            print("! - AUC: {:.3f},\nMCC: {:.3f}\n".format(results[5], results[4]))

    def _get_batch_by_id(self, batch_id, dataset_batched):
        """
        Retrieves a specific batch from a dataset based on its batch ID.

        Args:
            batch_id: int
                The ID of the batch to retrieve.
            dataset_batched: Dataset
                The batched dataset.

        Returns:
            batch: Tensor
                The batch with the specified ID.

        Raises:
            ValueError
                If the specified batch ID is not found in the dataset.
        """
        # Convert the dataset into an iterator
        test_iterator = iter(dataset_batched)

        # extract the batch with the batch_id from the test_dataset
        i = 0
        while i <= batch_id:
            try:
                batch = next(test_iterator)
            except StopIteration:
                raise ValueError(
                    f"Batch {batch_id} not found. The dataset has fewer batches."
                )
            if i == batch_id:
                return batch
            i += 1

    def _load_batched_data(self, batch_id):
        """
        Load one batch of data based on the specified batch ID.

        Args:
            batch_id: int
                The ID of the batch to load.

        Returns:
            one_batch: Batch
                The loaded batch of data.
        """
        if self.xai_config.integrated_gradients.dataset == "test":
            print("! - Loading one batch from test dataset")
            self.one_batch = self._get_batch_by_id(
                batch_id, self.test_dataset_batchwrapfetch
            )
        elif self.xai_config.integrated_gradients.dataset == "val":
            print("! - Loading one batch from validation dataset")
            self.one_batch = self._get_batch_by_id(
                batch_id, self.val_dataset_batchwrapfetch
            )
        elif self.xai_config.integrated_gradients.dataset == "train":
            print("! - Loading one batch from training dataset")
            self.one_batch = self._get_batch_by_id(
                batch_id, self.train_dataset_batchwrapfetch
            )

        return self.one_batch

    def _extract_data(self):
        """
        Extracts the dataset from the batch and returns the features, anomalous time series, and sample indices.

        Returns:
            features (list): The features extracted from the batch.
            anom_ts (list): The anomalous time series extracted from the batch.
            sample_indices (list): The sample indices extracted from the batch.
        """
        # extract the dataset from the batch
        self.features = self._get_nested_list_element(
            DATA_INDEX_BATCH["features"], self.one_batch
        )
        self.anom_ts = self._get_nested_list_element(
            DATA_INDEX_BATCH["anomalous_ts"], self.one_batch
        )
        # extract the sample indices
        self.sample_indices = self._get_nested_list_element(
            DATA_INDEX_BATCH["sample_indices"], self.one_batch
        )

        # if self.preproc_config.ds_type == "cml":
        #     self.sensor_lat_a = self.one_batch[0][2]
        #     self.sensor_lat_b = self.one_batch[0][3]
        #     self.sensor_lon_a = self.one_batch[0][4]
        #     self.sensor_lon_b = self.one_batch[0][5]
        # elif self.preproc_config.ds_type == "soilnet":
        #     self.sensor_lat = self.one_batch[0][2]
        #     self.sensor_lat = self.one_batch[0][3]

        return self.features, self.anom_ts, self.sample_indices

    def _load_batched_data_plot(self, batch_id):
        """
        Loads a batch of data for plotting.

        Args:
            batch_id (int): The ID of the batch to load.

        Returns:
            one_batch_plot (numpy.ndarray): The loaded batch of data.
        """
        # load one batch to memory
        # ------------------------------------------------------------------------------
        if self.xai_config.integrated_gradients.dataset == "test":
            self.one_batch_plot = self._get_batch_by_id(
                batch_id, self.test_dataset_batchwrapfetchplot
            )
        elif self.xai_config.integrated_gradients.dataset == "val":
            self.one_batch_plot = self._get_batch_by_id(
                batch_id, self.val_dataset_batchwrapfetchplot
            )
        elif self.xai_config.integrated_gradients.dataset == "train":
            self.one_batch_plot = self._get_batch_by_id(
                batch_id, self.train_dataset_batchwrapfetchplot
            )

        return self.one_batch_plot

    def _extract_data_plot(self):
        """
        Extracts data from the plotting batch.

        Returns:
            Tuple: A tuple containing the extracted data in the following order:
                - features_plot (list): The features from the plotting batch.
                - anom_ts_plot (list): The anomalous time series from the plotting batch.
                - sensor_ids (list): The sensor IDs from the plotting batch.
                - anomaly_dates (list): The anomaly dates from the plotting batch.
                - anomaly_flags_true (list): The true anomaly flags from the plotting batch.
        """
        # extract data from the plotting batch
        self.sensor_ids, self.anomaly_dates, self.anomaly_flags_true = (
            self._get_nested_list_element(
                DATA_INDEX_BATCH_PLOT["batch_anomaly_sensor_id"], self.one_batch_plot
            ),
            self._get_nested_list_element(
                DATA_INDEX_BATCH_PLOT["batch_dates"], self.one_batch_plot
            ),
            self._get_nested_list_element(
                DATA_INDEX_BATCH_PLOT["anomaly_flags"], self.one_batch_plot
            ),
        )
        
        self.features_plot = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["features"], self.one_batch_plot
        )
        self.anom_ts_plot = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["anomalous_ts"], self.one_batch_plot
        )

        self.lat_a = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["lat_a"], self.one_batch_plot
        )
        self.lat_b = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["lat_b"], self.one_batch_plot
        )
        self.lon_a = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["lon_a"], self.one_batch_plot
        )
        self.lon_b = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["lon_b"], self.one_batch_plot
        )
        self.sample_indices_plot = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["sample_indices"], self.one_batch_plot
        )
        self.distances = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["distances"], self.one_batch_plot
        )

        self.sensor_ind = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["sensor_ind"], self.one_batch_plot
        )

        self.batch_anomaly_sensor_id = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["batch_anomaly_sensor_id"], self.one_batch_plot
        )

        return (
            self.features_plot,
            self.anom_ts_plot,
            self.sensor_ids,
            self.anomaly_dates,
            self.anomaly_flags_true,
        )

    ############################################################################
    # IG METHODS
    ############################################################################
    def _zero_baseline(self, array):
        return tf.zeros(shape=array.shape)

    def _set_baseline(self, data_element, which=None):
        if which == None:
            which = self.xai_config.integrated_gradients.baseline

        # establish a baseline based on the shape of the data element
        if which == "zero":
            baseline = self._zero_baseline(data_element)
        elif which == "mean":
            raise NotImplementedError("Mean baseline not implemented yet.")
        elif which == "random":
            raise NotImplementedError("Random baseline not implemented yet.")
        else:
            raise NotImplementedError(
                "'Which' must be either 'mean' or 'zero' or 'random'."
            )

        return baseline

    def _interpolate(self, data_element, baseline):
        # linear interpoltaion
        m_steps = self.xai_config.integrated_gradients.m_steps
        self.alphas = tf.linspace(
            start=0.0, stop=1.0, num=m_steps + 1
        )  # Generate m_steps intervals for integral_approximation() below.

        # get the interpolated element depending on the shape of the element
        alphas_x = self.alphas[:, tf.newaxis, tf.newaxis]
        if len(data_element.shape) == 3:
            alphas_x = alphas_x[:, tf.newaxis]

        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(data_element, axis=0)
        delta = input_x - baseline_x
        data_element_series = baseline_x + alphas_x * delta

        if (
            self.xai_config.integrated_gradients.plot_interpolated_data_element_series
        ):
            self._plot_interpolated_data_element_series(
                data_element_series, self.alphas
            )
        return data_element_series

    def _mask_features(self, sample, features_series, anom_ts_series):
        pass
        # set all values of the features_series and anom_ts_series to zero
        # except for the given sample

        # THIS IS SOME EXAMPLE CODE HERE
        # indices = tf.range(10000)[:, tf.newaxis]
        # updates = tf.zeros((10000, 2), dtype=step.dtype)
        # step_masked = tf.tensor_scatter_nd_update(step, indices, updates)

    # @tf.function
    def compute_gradients(self, features_series, anom_ts_series, one_batch):
        ## ToDo
        # - find a way to only compute the gradients for a certain sample and not for the whole batch
        path_gradients_features = []
        path_gradients_anom_ts = []
        path_predictions = []

        for i in range(features_series.shape[0]):
            print(f"! - Computing gradients for interpolation step {i}")
            step_features = features_series[i]
            step_anom_ts = anom_ts_series[i]

            with tf.GradientTape(persistent=True) as tape:
                # watch the gradients for just one element in the input dataset
                tape.watch(step_features)
                tape.watch(step_anom_ts)
                # predict on the full batch since only entire batches can be fed to the model
                one_batch_list = self._tuple_to_list(one_batch[0])
                # replace the data with the interpolated step
                one_batch_list[0] = step_features
                one_batch_list[1] = step_anom_ts
                one_batch_tuple = self._list_to_tuple(one_batch_list)
                self.one_batch_tuple = one_batch_tuple
                # predict on the batch with inserted step
                # USE THIS LINE FOR LOCAL TESTING
                #prediction = self.model(one_batch_tuple[:-1])
                # USE THIS LINE ON THE EVE CLUSTER
                prediction = self.model(one_batch_tuple)
                path_predictions.append(prediction)

            # calcuate the gradients for the interpolated step with respect to the corresponding prediction
            gradient_features_series = tape.gradient(prediction, step_features)
            gradient_anom_ts_series = tape.gradient(prediction, step_anom_ts)
            print(f"! - Gradients computed.")
            # print("Gradients shape: {}".format(gradient_.shape))
            # print("Gradient type: {}".format(type(gradient)))
            # print("Gradient dtype: {}".format(gradient.dtype))
            path_gradients_features.append(gradient_features_series)
            path_gradients_anom_ts.append(gradient_anom_ts_series)
            print("##############################################")
        # convert the list to a tensor
        self.path_gradients_features = tf.convert_to_tensor(path_gradients_features)
        self.path_gradients_anom_ts = tf.convert_to_tensor(path_gradients_anom_ts)
        self.path_predictions = tf.convert_to_tensor(path_predictions)

        return (
            self.path_gradients_features,
            self.path_gradients_anom_ts,
            self.path_predictions,
        )

    def integral_approximation(self, path_gradients_unwrapped):
        # riemann_trapezoidal
        grads = (
            path_gradients_unwrapped[:-1] + path_gradients_unwrapped[1:]
        ) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    ############################################################################
    # SAMPLE UNWRAPPING
    ############################################################################
    def _unwrap_features(self, index, features):
        # Shape (n_timesteps, n_neighbors, n_features)
        series = []
        mask = self.sample_indices == index
        a = features[mask]
        timesteps = (
            self.preproc_config.timestep_before + self.preproc_config.timestep_after + 1
        )
        n_neighbors = int(len(a) / timesteps)
        for i in range(n_neighbors):
            series.append(a[i::n_neighbors])
        # shape(neighbor, timesteps, features)
        features_unwrapped = tf.stack(series, axis=0)
        return features_unwrapped

    def _unwrap_anom_ts(self, index, anom_ts):
        return anom_ts[index]

    def _unwrap_gradients_features(self, index, gradients_features):
        return self._unwrap_features(index, gradients_features)

    def _unwrap_gradients_anom_ts(self, index, gradients_anom_ts):
        return gradients_anom_ts[index]

    def _unwrap_predictions(self, index, predictions):
        return predictions[index]

    def _unwrap_all(self, index):
        self.features_unwrapped = self._unwrap_features(index, self.features)
        self.anom_ts_unwrapped = self._unwrap_anom_ts(index, self.anom_ts)
        
        
        self.prediction_unwrapped = self._unwrap_predictions(
            index, self.anomaly_flags_pred
        )
        self.anomaly_flag_pred_class_unwrapped = self._unwrap_predictions(
            index, self.anomaly_flags_pred_class
        )
        self.anomaly_flag_true_unwrapped = self._unwrap_predictions(
            index, self.anomaly_flags_true
        )
        
        if self.xai_config.integrated_gradients.load_gradients_from_sample_file == False:
            self.gradients_features_unwrapped = self._unwrap_gradients_features(
                index, self.gradients_features
            )
            self.gradients_anom_ts_unwrapped = self._unwrap_gradients_anom_ts(
                index, self.gradients_anom_ts
            )

        # TO DO
        # if self.preproc_config.ds_type == 'cml':
        #     self.sensor_lat_a_unwrapped = self.ASDASDASD(
        #         index, self.sensor_lat_a
        #     )
        #     self.sensor_lat_b_unwrapped = self.ASDASDASD(
        #         index, self.sensor_lat_b
        #     )
        #     self.sensor_lon_a_unwrapped = self.ASDASDASD(
        #         index, self.sensor_lon_a
        #     )
        #     self.sensor_lon_b_unwrapped = self.ASDASDASD(
        #         index, self.sensor_lon_b
        #     )

        # elif self.preproc_config.ds_type == 'soilnet':
        #     self.sensor_lat_unwrapped = self.ASDASDASD(
        #         index, self.sensor_lat
        #     )
        #     self.sensor_lon_unwrapped = self.ASDASDASD(
        #         index, self.sensor_lon
        #     )

    ############################################################################
    # IG EXECUTION
    ############################################################################
    def get_gradients(self):
        """
        Calculate the gradients for the integrated gradients method.
        Iterates over the batches and calculates the gradients for each batch
        seperately but for all samples at once. Then the gradients are unwrapped
        and plotted for each sample.

        Args:
            unwrap_and_plot (bool, optional): Whether to unwrap and plot the gradients. Defaults to True.

        Raises:
            ValueError: If batch_ids is not a list or 'all'.

        Returns:
            None
        """
        if self.xai_config.integrated_gradients.batch_ids == "all":
            # call self._get_gradients_all_batches() as long as batches are available
            # so until raise ValueError(f"Batch {batch_id} not found. The dataset has fewer batches.")
            # from self.load_batched_data() is raised
            batch_id = 0
            while True:
                try:
                    self._get_gradients_single_batch(batch_id)
                    batch_id += 1
                except ValueError:
                    break
        # check if batch_id is a list, then iterate over the list
        elif isinstance(
            self.xai_config.integrated_gradients.batch_ids, oc.listconfig.ListConfig
        ):
            for batch_id in self.xai_config.integrated_gradients.batch_ids:
                self._get_gradients_single_batch(batch_id)
        else:
            raise ValueError(
                f"batch_ids must be a list or 'all', but is {self.xai_config.integrated_gradients.batch_ids}"
            )

        print("! - Done.")

    def _get_gradients_single_batch(self, batch_id):
        print("#################################################")
        print(f"! - Loading batch {batch_id}")
        print("#################################################")
        self.current_batch_id = batch_id
        self.one_batch = self._load_batched_data(self.current_batch_id)
        # loads the plotting batch form the data
        self._load_batched_data_plot(batch_id)
        # extracts the data from the plotting batch
        self._extract_data_plot()
        print("! - Extracting data from batch")
        features, anom_ts, sample_indices = self._extract_data()
        print("! - Setting baseline", self.xai_config.integrated_gradients.baseline)
        self.baseline_features = self._set_baseline(
            features, which=self.xai_config.integrated_gradients.baseline
        )
        self.baseline_anom_ts = self._set_baseline(
            anom_ts, which=self.xai_config.integrated_gradients.baseline
        )
        print("! - Interpolating")
        self.features_series = self._interpolate(features, self.baseline_features)
        self.anom_ts_series = self._interpolate(anom_ts, self.baseline_anom_ts)

        if not self.xai_config.integrated_gradients.load_gradients_from_sample_file:
            print("! - Computing gradients")
            self.compute_gradients(
                self.features_series, self.anom_ts_series, self.one_batch
            )
            # integrate gradients along the path
            self.gradients_features = self.integral_approximation(
                self.path_gradients_features
            )
            self.gradients_anom_ts = self.integral_approximation(
                self.path_gradients_anom_ts
            )
            # if self.xai_config.integrated_gradients.save_unscaled_gradients_batch == True:
            #     # save the gradients and predictions
            #     np.save(
            #         Path(
            #             self.output_dir,
            #             f"gradients_feature_batch_{self.current_batch_id}.npy",
            #         ),
            #         self.gradients_features,
            #     )
            #     np.save(
            #         Path(
            #             self.output_dir,
            #             f"gradients_anom_ts_batch_{self.current_batch_id}.npy",
            #         ),
            #         self.gradients_anom_ts,
            #     )
            # scale the gradients with respect to the input data
            if self.xai_config.integrated_gradients.scale_gradients == True:
                self.gradients_features = (
                    self.features - self.baseline_features
                ) * self.gradients_features
                self.gradients_anom_ts = (
                    self.anom_ts - self.baseline_anom_ts
                ) * self.gradients_anom_ts
            # treat negative values
            if self.xai_config.integrated_gradients.negative_values == "abs":
                self.gradients_features = tf.abs(self.gradients_features)
                self.gradients_anom_ts = tf.abs(self.gradients_anom_ts)
            elif self.xai_config.integrated_gradients.negative_values == "clip":
                # remove all negaitve values and set them to 0
                self.gradients_features = tf.clip_by_value(
                    self.gradients_features, 0, np.inf
                )
                self.gradients_anom_ts = tf.clip_by_value(self.gradients_anom_ts, 0, np.inf)
            elif self.xai_config.integrated_gradients.negative_values == "keep":
                pass
            else:
                raise ValueError(
                    f"config.integrated_gradients.negative_values must be 'abs' or 'clip', but is {self.xai_config.integrated_gradients.absolute_values}"
                )
            # # save the scaled gradients
            # if self.xai_config.integrated_gradients.save_scaled_gradients_batch == True:
            #     np.save(
            #         Path(
            #             self.output_dir,
            #             f"gradients_scaled_feature_batch_{self.current_batch_id}.npy",
            #         ),
            #         self.path_gradients_features,
            #     )
            #     np.save(
            #         Path(
            #             self.output_dir,
            #             f"gradients_scaled_anom_ts_batch_{self.current_batch_id}.npy",
            #         ),
            #         self.path_gradients_anom_ts,
            #     )
        
        # get the predictions for the batch
        # USE THIS LINE FOR LOCAL TESTING
        #self.anomaly_flags_pred = self.model(self.one_batch[0][:-1], training=False)
        # USE THIS LINE ON THE EVE CLUSTER
        self.anomaly_flags_pred = self.model(self.one_batch[0], training=False)
        self.anomaly_flags_pred_class = np.where(
            self.anomaly_flags_pred > self.xai_config.integrated_gradients.threshold,
            1,
            0,
        )
        self.anomaly_flags_true = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["anomaly_flags"], self.one_batch_plot
        )

        # if self.xai_config.integrated_gradients.save_predictions_batch == True:
        #     np.save(
        #         Path(self.output_dir, f"predictions_batch_{self.current_batch_id}.npy"),
        #         self.anomaly_flags_pred,
        #     )

        # unwrap the gradients, data and continue with plotting
        self._unwrap_and_save()

    def _unwrap_and_save(self):
        """
        Unwraps the gradients and data for selected samples and plots the results.
        """
        self.sample_indexes = self._get_confusion_matrix_indexes(
            self.anomaly_flags_true,
            self.anomaly_flags_pred_class,
            self.xai_config.integrated_gradients.which_samples,
        )

        print(
            "! - The following Sample indexes remained after selecting:",
            self.sample_indexes,
        )

        for sample_index in self.sample_indexes:
            # get some informatioin about the current sample
            (
                self.current_sensor_id,
                self.current_anomaly_dates,
                self.current_anomaly_date,
                self.current_anomaly_flag_true,
            ) = self._get_current_target_info(sample_index)
            # create output directory and file name
            self._current_file_name = self._get_file_name(
                self.xai_config.project,
                self.preproc_config.ds_type,
                self.xai_config.integrated_gradients.dataset,
                self.current_sensor_id,
                self.current_anomaly_date,
                int(self.anomaly_flags_true[sample_index]),
                int(self.anomaly_flags_pred_class[sample_index]),
            )
            # create an output directory
            self._current_output_dir = self._create_output_dir(
                self.xai_config.output_dir,
                self.xai_config.project,
                self.preproc_config.ds_type,
                self.xai_config.integrated_gradients.dataset,
                self.current_sensor_id,
                self.current_anomaly_date,
                int(self.anomaly_flags_true[sample_index]),
                int(self.anomaly_flags_pred_class[sample_index]),
            )
            # unwrap all elements for the current sample
            print(
                f"! - Unwrapping and plotting batch {self.current_batch_id} for sample index {sample_index} with sensor id {self.current_sensor_id} and anomaly date {self.current_anomaly_date}"
            )

            # get the distances of the neighboring sensors to anomalous sensor
            distances = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["batch_dates"], self.one_batch_plot
        )


            self._unwrap_all(sample_index)

            # load unwrapped gradients from file
            if self.xai_config.integrated_gradients.load_gradients_from_sample_file: 
                print("! - Loading gradients from file")               
                try:
                    self.gradients_features_unwrapped = np.load(
                        Path(
                            self._current_output_dir,
                            "gradients_features_unwrapped_" + self._current_file_name + ".npy",
                        )
                    )
                    self.gradients_anom_ts_unwrapped = np.load(
                        Path(
                            self._current_output_dir,
                            "gradients_anom_ts_unwrapped_" + self._current_file_name + ".npy",
                        )
                    )
                except FileNotFoundError:
                    print("! - File for sample {} not found. Skipping sample.".format(sample_index))
                    continue
            
            else:
                np.save(
                    Path(
                        self._current_output_dir,
                        "gradients_features_unwrapped_" + self._current_file_name + ".npy",
                    ),
                    self.gradients_features_unwrapped,
                )
                np.save(
                    Path(
                        self._current_output_dir,
                        "gradients_anom_ts_unwrapped_" + self._current_file_name + ".npy",
                    ),
                    self.gradients_anom_ts_unwrapped,
                )

            
            # save unwrapped data to disc
            np.save(
                Path(
                    self._current_output_dir,
                    "features_unwrapped_" + self._current_file_name + ".npy",
                ),
                self.features_unwrapped,
            )
            np.save(
                Path(
                    self._current_output_dir,
                    "anom_ts_unwrapped_" + self._current_file_name + ".npy",
                ),
                self.anom_ts_unwrapped,
            )
            np.save(
                Path(
                    self._current_output_dir,
                    "predictions_unwrapped_" + self._current_file_name + ".npy",
                ),
                self.prediction_unwrapped,
            )
            np.save(
                Path(
                    self._current_output_dir,
                    "anomaly_flag_true_unwrapped_" + self._current_file_name + ".npy",
                ),
                self.anomaly_flag_true_unwrapped,
            )
            
            if self.xai_config.integrated_gradients.plot_classified_timeseries_sample:
                # plot the classified time series for the sample
                self.plot_classified_timeseries_sample(sample_index, predictions=None)

            if not self.xai_config.integrated_gradients.load_gradients_from_sample_file:
                if self.xai_config.integrated_gradients.plot_gradient_saturation:
                    print("! - Plotting gradient saturation")
                    self._plot_gradient_saturation(sample_index)

            self.log_file(
                self.xai_config.output_dir,
                self.xai_config.project,
                self.preproc_config.ds_type,
                self.xai_config.integrated_gradients.dataset,
                self.current_batch_id,
                sample_index,
                self.current_sensor_id,
                self.current_anomaly_date,
            )

        if self.xai_config.integrated_gradients.plot_heatmap:
            print("! - Plotting heatmap")
            self._plot_ig_heatmap(
                features_unwrapped = self.features_unwrapped,
                anom_ts_unwrapped = self.anom_ts_unwrapped,
                gradients_features_unwrapped = self.gradients_features_unwrapped,
                gradients_anom_ts_unwrapped = self.gradients_anom_ts_unwrapped,
                anomaly_flag_true_unwrapped = self.anomaly_flag_true_unwrapped,
                anomaly_flag_pred_class_unwrapped = self.anomaly_flag_pred_class_unwrapped,
                prediction_unwrapped = self.prediction_unwrapped,
                current_sensor_id = self.current_sensor_id,
                current_anomaly_date = self.current_anomaly_date,
                _current_output_dir = self._current_output_dir,
                _current_file_name = self._current_file_name,
                annotate_batch_id = self.xai_config.integrated_gradients.plot.heatmap.annotate_batch_id,
                scale_feature_gradients=self.xai_config.integrated_gradients.plot.heatmap.scale_feature_gradients,
                dpi = self.xai_config.integrated_gradients.plot.heatmap.dpi,
            )


    ############################################################################
    # VISUALIZATION
    ############################################################################
    def _plot_interpolated_data_element_series(self, data_element_series, alphas):
        """
        Plot the interpolated data element series.

        Args:
            data_element_series (ndarray): The data element series to plot.
            alphas (ndarray): The alpha values for interpolation.

        Returns:
            None
        """
        fig = plt.figure(figsize=(20, 10))
        i = 0
        if len(data_element_series.shape) == 3:
            ymax = np.max(data_element_series)
            ymin = np.min(data_element_series)
            for alpha, data_element_ in zip(alphas[0::10], data_element_series[0::10]):
                i += 1
                plt.subplot(len(alphas[0::10]), 1, i)
                plt.title(f"alpha: {alpha:.1f}")
                plt.plot(data_element_[:500, :])
                plt.ylim(ymin, ymax)
            plt.tight_layout()
            plt.savefig(
                Path(
                    self.output_dir,
                    f"interpolated_data_element_1_batch_{self.current_batch_id}.png",
                ),
                dpi=50,
            )
            plt.close("all")

        if len(data_element_series.shape) == 4:
            ymax = np.max(data_element_series[:, 0, :, :])
            ymin = np.min(data_element_series[:, 0, :, :])
            for alpha, data_element_ in zip(alphas[0::10], data_element_series[0::10]):
                i += 1
                plt.subplot(len(alphas[0::10]), 1, i)
                plt.title(f"alpha: {alpha:.1f}")
                plt.plot(data_element_[0, :, :])
                plt.ylim(ymin, ymax)

            plt.ylim(ymin, ymax)
            plt.tight_layout()
            plt.savefig(
                Path(
                    self.output_dir,
                    f"interpolated_data_element_2_batch_{self.current_batch_id}.png",
                ),
                dpi=50,
            )
        plt.close("all")

    def plot_classified_timeseries_sample(self, sample_index, predictions=None):
        """
        Plots the timeseries for a specific sample in the batch.

        Args:
            sample_index (int): The index of the sample in the batch.
            predictions (optional): Predictions for the sample. Defaults to None.

        Returns:
            None
        """
        # extract the data from the batch
        sensor_dates = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["batch_dates"], self.one_batch_plot
        )
        sensor_ids = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["batch_anomaly_sensor_id"], self.one_batch_plot
        )
        sensor_anomaly_timeseries = self._get_nested_list_element(
            DATA_INDEX_BATCH_PLOT["anomalous_ts"], self.one_batch_plot
        )

        curr_dates = pd.to_datetime(
            [x.decode("utf-8") for x in tf.squeeze(sensor_dates[sample_index]).numpy()],
            format="%Y-%m-%dT%H:%M:%S",
        )

        curr_sensor_id = tf.squeeze(sensor_ids[sample_index]).numpy().decode("utf-8")

        out_dir = os.path.join(
            self._current_output_dir, "anomalous_ts_" + self._current_file_name + ".png"
        )
        anomaly_time_ind = self.model.model_info[0].numpy()
        model_config = self.model_config
        # plot the timeseries for certain sample
        timeseries_figure(
            self.anomaly_flags_pred_class[sample_index][0],
            self.anomaly_flags_true[sample_index],
            sensor_anomaly_timeseries[sample_index],
            curr_sensor_id,
            curr_dates,
            out_dir,
            anomaly_time_ind,
            model_config,
            predictions=predictions,
        )
        plt.close("all")

    def _plot_gradient_saturation(self, sample_index):
        """
        Plots the gradient saturation for a given sample index.

        Args:
            sample_index (int): The index of the sample to plot the gradient saturation for.

        Returns:
            None
        """
        average_gradients_features = []
        average_gradients_anom_ts = []
        step_predictions_unwrapped = []

        # iterate over the path and take the mean of each interpolation step
        for step_gradients_feature, step_gradients_anom_ts, step_predictions in zip(
            self.path_gradients_features,
            self.path_gradients_anom_ts,
            self.path_predictions,
        ):
            step_gradients_features_unwrapped = self._unwrap_features(
                sample_index, step_gradients_feature
            )
            step_gradients_anom_ts_unwrapped = self._unwrap_anom_ts(
                sample_index, step_gradients_anom_ts
            )
            step_prediction_unwrapped = self._unwrap_predictions(
                sample_index, step_predictions
            )
            # Get average value for each interpolation step averaged over time and neighbors
            average_gradient_features_unwrapped = tf.reduce_mean(
                step_gradients_features_unwrapped
            )
            average_gradient_anom_ts_unwrapped = tf.reduce_mean(
                step_gradients_anom_ts_unwrapped
            )
            average_gradients_features.append(average_gradient_features_unwrapped)
            average_gradients_anom_ts.append(average_gradient_anom_ts_unwrapped)
            step_predictions_unwrapped.append(step_prediction_unwrapped)
        # convert to tensor
        average_gradients_features = tf.stack(average_gradients_features)
        average_gradients_anom_ts = tf.stack(average_gradients_anom_ts)
        step_predictions_unwrapped = tf.stack(step_predictions_unwrapped)

        if self.xai_config.integrated_gradients.plot.gradient_saturation.normalize:
            # Normalize gradients to 0 to 1 scale. E.g. (x - min(x))/(max(x)-min(x))
            average_gradients_features = (
                average_gradients_features
                - tf.math.reduce_min(average_gradients_features, axis=0)
            ) / (
                tf.math.reduce_max(average_gradients_features, axis=0)
                - tf.reduce_min(average_gradients_features, axis=0)
            )
            average_gradients_anom_ts = (
                average_gradients_anom_ts
                - tf.math.reduce_min(average_gradients_anom_ts, axis=0)
            ) / (
                tf.math.reduce_max(average_gradients_anom_ts, axis=0)
                - tf.reduce_min(average_gradients_anom_ts, axis=0)
            )

        plt.figure(figsize=(10, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(self.alphas, step_predictions_unwrapped)
        ax1.set_title("Target class predicted \n probability over alpha")
        ax1.set_ylabel("model predicted probability")
        ax1.set_xlabel("alpha")

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(self.alphas, average_gradients_features, label="Features")
        ax2.set_title("Average timestep gradients \n (normalized) over alpha")
        ax2.set_ylabel("Average timestep gradients features")
        ax2.set_xlabel("alpha")
        ax2.legend(loc="upper left")

        ax2_2 = ax2.twinx()
        ax2_2.plot(
            self.alphas,
            average_gradients_anom_ts,
            color="orange",
            label="Anom. TS",
        )
        ax2_2.set_ylabel("Average timestep gradients anomalous time series")
        # position the legend upper right
        ax2_2.legend(loc="upper right")

        plt.tight_layout()

        plt.savefig(
            Path(
                self._current_output_dir,
                "gradient_saturation_" + self._current_file_name + ".png",
            )
        )
        plt.close("all")

    def _plot_ig_heatmap(
        self,
        features_unwrapped,
        anom_ts_unwrapped,
        gradients_features_unwrapped,
        gradients_anom_ts_unwrapped,
        anomaly_flag_true_unwrapped,
        anomaly_flag_pred_class_unwrapped,
        prediction_unwrapped,
        current_sensor_id,
        current_anomaly_date,
        _current_output_dir,
        _current_file_name,
        annotate_batch_id=True,
        scale_feature_gradients=1,
        dpi=100,
    ):
        """
        Plot the integrated gradients heatmap.

        This method plots the integrated gradients heatmap for the given time series data and gradients.
        It visualizes the time series data along with the corresponding gradients as a heatmap.

        Returns:
            None
        """
        n_timesteps = (
            self.preproc_config.timestep_before + self.preproc_config.timestep_after + 1
        )
        # bring time series axis to the front
        # stack all time series data from all features
        features = tf.transpose(features_unwrapped, perm=[1, 0, 2])
        features = tf.reshape(features, (n_timesteps, -1))
        anom_ts = tf.reshape(anom_ts_unwrapped, (n_timesteps, 2))
        data_time_series = tf.concat([anom_ts, features], axis=1)
        # stack all gradients data from all features
        # grad_features = tf.reshape(self.gradients_features_unwrapped, (n_timesteps, 6))
        grad_features = tf.transpose(gradients_features_unwrapped, perm=[1, 0, 2])
        grad_features = tf.reshape(grad_features, (n_timesteps, -1))
        if scale_feature_gradients == 'auto':
            # scale the gradients with the number of neighbors
            grad_features = grad_features * grad_features.shape[1]
        else:    
            grad_features = grad_features * scale_feature_gradients
        grad_anom_ts = tf.reshape(gradients_anom_ts_unwrapped, (n_timesteps, 2))
        data_gradients = tf.concat([grad_anom_ts, grad_features], axis=1)
        n_series = data_time_series.shape[1]
        

        # FIGURE CREATION
        # ----------------
        gs_top = plt.GridSpec(
            n_series,
            1,
            top=0.9,
            bottom=0.17,
            right=0.8,
            left=0.2,
        )
        gs_mid = plt.GridSpec(
            n_series,
            1,
            top=0.82,
            bottom=0.17,
            right=0.8,
            left=0.2,
        )

        gs_base = plt.GridSpec(
            n_series,
            1,
            top=0.70,
            bottom = 0.17,
            right=0.8,
            left=0.2,
        )

        if self.xai_config.integrated_gradients.plot.heatmap.ylims == "auto":
            # get min max for yscale
            ymin = tf.reduce_min(data_time_series)
            ymax = tf.reduce_max(data_time_series)
        else:
            ymin = self.xai_config.integrated_gradients.plot.heatmap.ylims[0]
            ymax = self.xai_config.integrated_gradients.plot.heatmap.ylims[1]
        
        if ymax > 15:
            ymax = 15
        
        timestep_prediction = self.preproc_config.timestep_before + 1

        # get min max for colorbar
        if (
            not self.xai_config.integrated_gradients.plot.heatmap.cbar_limits
            == "auto"
        ):
            vmin = self.xai_config.integrated_gradients.plot.heatmap.cbar_limits[0]
            vmax = self.xai_config.integrated_gradients.plot.heatmap.cbar_limits[1]
        else:
            vmin = tf.reduce_min(data_gradients)
            vmax = tf.reduce_max(data_gradients)
        # TODO
        # make logic of colormap and norm more flexible and appropriate for keywords
        if self.xai_config.integrated_gradients.negative_values == "keep":
            colormap = "RdBu_r"

            # if np.abs(data_gradients).min() == 0:
            #     linthresh = vmin
            # else:
            #     linthresh = np.abs(data_gradients).min()
            # norm = SymLogNorm(linthresh=linthresh, linscale=1, vmin=-vmax, vmax=vmax)

        else:
            colormap = "Blues"
            if (
                self.xai_config.integrated_gradients.plot.heatmap.cbar_norm
                == "log"
            ):
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        # FIGURE CREATION
        # ----------------
        fig = plt.figure(figsize=(8 * 0.8, (n_series + 2) * 0.7 ), constrained_layout=True)
        # fig, axes = plt.subplots(
        #     nrows=n_series + 1, ncols=1, figsize=(8 * 0.7, (n_series + 2) * 0.7 ), sharex=True
        # )
        axis = []
        if self.xai_config.integrated_gradients.plot.heatmap.cbar_norm == "linear":
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        for i in range(n_series):
            if i == 0 or i == 1:
                ax = fig.add_subplot(gs_top[i])
                axis.append(ax)
            elif i == 2 or i == 3:
                ax = fig.add_subplot(gs_mid[i])
                axis.append(ax)
            else:
                ax = fig.add_subplot(gs_base[i])
                axis.append(ax)
            ax.plot(data_time_series[:, i], c="red", label="Time Series")
            # use the values of data_gradients as background color
            ts_length = data_gradients.shape[0]
            x = np.linspace(-0.5, ts_length - 0.5, ts_length + 1)
            y = [ymin, ymax]
            z = data_gradients[:, i]
            Y = y
            Z = np.reshape(z, (1, ts_length))
            X = x
            pcol = ax.pcolormesh(X, Y, Z, norm=norm, alpha=0.8, cmap=colormap)

            # set the ylimits and labels for all plots
            if i == 0:
                ax.set_ylabel("TL 1",rotation=0, labelpad=30)
            elif i == 1: 
                ax.set_ylabel("TL 2",rotation=0, labelpad=30)
            elif i == 2:
                ax.set_ylabel("N 0, TL 1",rotation=0, labelpad=30)
            elif i == 3:
                ax.set_ylabel("N 0, TL 2",rotation=0, labelpad=30)
            if i == 0:
                ax.set_title("Flagged Sensor")
            elif i == 2:
                ax.set_title("Self reference cycle input in graph layout")
            elif i == 4:
                ax.set_title("Neighboring sensor input in graph layout")

            if i > 3:
                if i % 2 == 0:
                    ax.set_ylabel(f"N {i // 2 - 1}, TL 1",rotation=0, labelpad=30)
                else:
                    ax.set_ylabel(f"N {i // 2 - 1}, TL 2",rotation=0, labelpad=30)
            ax.vlines(
                [timestep_prediction - 0.5, timestep_prediction + 0.5],
                ymin=ymin,
                ymax=ymax,
                colors="k",
                label="Prediction",
            )

        
        # set a shared title for the whole figure
        fig.suptitle(
            "Sensor " + current_sensor_id + ", " + str(current_anomaly_date),
            fontsize=12,
        )


        # select the color according to the colormap
        color = COLOR_MAP[
            (
                int(anomaly_flag_true_unwrapped),
                int(anomaly_flag_pred_class_unwrapped),
            )
        ]
        # axes[n_series].bar(
        #     x=timestep_prediction,
        #     height=prediction_unwrapped,
        #     width=1,
        #     color=color,
        #     label="Prediction",
        # )
        # axes[n_series].set_ylim([0, 1])
        # # integrate
        # # set the ylimits for all plots
        # for i in range(n_series):
        #     axes[i].set_ylim([ymin, ymax])

        # set figure title
        # fig.suptitle(
        #     "Sensor " + current_sensor_id + ", " + str(current_anomaly_date),
        #     fontsize=16,
        # )

        # colorbar
        # Define the position and size of the colorbar
        cbar_width_percentage = 50  # Percentage of the width the colorbar should span
        cbar_height = 0.2  # Height of the colorbar in inches
        cbar_x = (100 - cbar_width_percentage) / 2  # X-position of the bottom-left corner of the colorbar
        cbar_y = 0.65  # Y-position of the bottom-left corner of the colorbar

        # Convert percentages to relative coordinates
        fig_width_inches = fig.get_size_inches()[0]
        fig_height_inches = fig.get_size_inches()[1]
        cbar_x_rel = cbar_x / 100
        cbar_y_rel = cbar_y / fig_height_inches
        cbar_width_rel = cbar_width_percentage / 100
        cbar_height_rel = cbar_height / fig_height_inches

        # Add the colorbar
        cbar_ax = fig.add_axes([cbar_x_rel, cbar_y_rel, cbar_width_rel, cbar_height_rel])
        cbar = fig.colorbar(pcol, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Attribution")
        
        # cbar_ax = fig.add_axes([0.2, -0.1, 0.8, 0.05])  # [x, y, width, height] left
        # cbar = fig.colorbar(pcol, cax=cbar_ax, orientation="horizontal")
        # cbar.set_label("Attribution")

        # make a stronger fat line around the first xnumber of subplots
        xnumber = anom_ts.shape[1]
        # format top
        for i in range(xnumber):
            axis[i].spines["top"].set_linewidth(2)
            axis[i].spines["right"].set_linewidth(2)
            axis[i].spines["left"].set_linewidth(2)
            axis[i].spines["bottom"].set_linewidth(2)
            # turn of labels
            axis[i].set_xticklabels([])
        # format mid
        for i in range(2,4):
            axis[i].set_xticklabels([])
        # format base
        for i in range(4, n_series-1):
            axis[i].set_xticklabels([])


        if annotate_batch_id:
            # print a little number in the bottom left corner of the figure
            try:
                text = "Batch ID: " + str(self.current_batch_id)
            except AttributeError:
                text = ''
                pass
            fig.text(0.1, 0.1, text, ha="center", va="center")
        
        # add the prediction as text below the batch ID
        #format as float with 2 decimals
        text = "Prediction: {:.2f}".format(prediction_unwrapped[0])
        fig.text(0.1, 0.07, text, ha="center", va="center")

        print("! - Saving heatmap to ", _current_output_dir)
        plt.savefig(
            Path(
                _current_output_dir,
                "ig_heatmap_" + _current_file_name + ".png",
            ), dpi=dpi,
        )
        
        plt.close("all")
        print("! - Heatmap saved.")



    def plot_ig_heatmap_from_directory(
        self, overwrite=False, sensors=None, time_from=None, time_to=None
    ): 
        """
        Execute _plot_ig_heatmap for all directories configured by xai_config.

        Args:
            new_heatmap (bool, optional): Whether to generate a new heatmap or not. Defaults to True.
            overwrite (bool, optional): Whether to overwrite existing heatmaps or not. Defaults to False.
        """
        if sensors == None:
            # generate a list of all sensors
            sensors = os.listdir(self.output_dir)
        # remove log form sensors
        sensors = [s for s in sensors if s != "log"]
        # remove .DS_Store from sensors
        sensors = [s for s in sensors if s != ".DS_Store"]

        print(f"Found {len(sensors)} sensors.")
        print(sensors)
        
        # get a list of all samples as subdirectories from the sensors
        samples = []
        for sensor in sensors:
            subdirs = os.listdir(os.path.join(self.output_dir, sensor))
            subdirs = [d for d in subdirs if os.path.isdir(os.path.join(self.output_dir, sensor, d))]
            samples.append(subdirs)
        samples = sorted(list(itertools.chain.from_iterable(samples)))

        # subset the samples based on the time_from and time_to
        if time_from is not None and time_to is not None:
            time_from = pd.to_datetime(time_from)
            time_to = pd.to_datetime(time_to)
            samples_select = []
            for sample in samples:
                anomaly_date = "_".join(sample.split("_")[4:6])
                anomaly_date = pd.to_datetime(anomaly_date, format="%Y%m%d_%H%M%S")
                if anomaly_date >= time_from and anomaly_date <= time_to:
                    samples_select.append(sample)
            samples = samples_select

        if self.workerid is not None:
            print("Detected array job.")
            print("Distributing work based on number of samples and workers.")
            print("Workerid: ", self.workerid)
            print("Number of workers: ", self.n_worker)
            samples = self._split_work(samples, self.workerid, self.n_worker)
            print("Number of directories assigned to worker: ", len(samples))

        # number of samples remainedn after splitting
        print("Number of directories to process: ", len(samples))

        # iterate over directories and plot the heatmap
        # load the features_unwrapped and ig_unwrapped from the npy file
        for sample in samples:
            # get the path to the npy file: features
            sensor = "_".join(sample.split("_")[:4])
            anomaly_date = "_".join(sample.split("_")[4:6])
            anomaly_date = pd.to_datetime(anomaly_date, format="%Y%m%d_%H%M%S")

            stem_file_name = (
                self.xai_config.project
                + "_"
                + self.preproc_config.ds_type
                + "_"
                + self.xai_config.integrated_gradients.dataset
                + "_"
                + sample
                + ".npy"
            )
            fn_features = "_".join(["features_unwrapped", stem_file_name])
            fp_features = os.path.join(self.output_dir, sensor, sample, fn_features)
            fn_anom_ts = "_".join(["anom_ts_unwrapped", stem_file_name])
            fp_anom_ts = os.path.join(self.output_dir, sensor, sample, fn_anom_ts)
            fn_gradients_features = "_".join(
                ["gradients_features_unwrapped", stem_file_name]
            )
            fp_gradients_features = os.path.join(
                self.output_dir, sensor, sample, fn_gradients_features
            )

            fn_gradients_anom_ts = "_".join(
                ["gradients_anom_ts_unwrapped", stem_file_name]
            )
            fp_gradients_anom_ts = os.path.join(self.output_dir, sensor, sample, fn_gradients_anom_ts)
            # fn_prediction_unwrapped = "_".join(
            #     ["predictions_unwrapped", stem_file_name[1:]]
            # )
            fn_prediction_unwrapped =  "predictions_unwrapped_" + stem_file_name

            fp_prediction_unwrapped = os.path.join(
                self.output_dir, sensor, sample, fn_prediction_unwrapped
            )
            # load the npy file
            try:
                features_unwrapped = np.load(fp_features, allow_pickle=True)
            except FileNotFoundError:
                print(f"! - File {fp_features} not found. Skipping sample.")
                continue
            anom_ts_unwrapped = np.load(fp_anom_ts, allow_pickle=True)
            gradients_features_unwrapped = np.load(
                fp_gradients_features, allow_pickle=True
            )
            gradients_anom_ts_unwrapped = np.load(
                fp_gradients_anom_ts, allow_pickle=True
            )
            anomaly_flag_true_unwrapped = sample[-3]
            anomaly_flag_pred_class_unwrapped = sample[-1]
            prediction_unwrapped = np.load(fp_prediction_unwrapped, allow_pickle=True)
            _current_output_dir = os.path.join(self.output_dir, sensor, sample)
            _current_file_name = stem_file_name[:-4]

            fn_heatmap = (
                "ig_heatmap_"
                + self.xai_config.project
                + "_"
                + self.xai_config.integrated_gradients.dataset
                + "_"

                + sensor
                + ".png"
            )

            fn_heatmap = "ig_heatmap_" +  '_'.join([
                            self.xai_config.project,
                            self.preproc_config.ds_type,
                            self.xai_config.integrated_gradients.dataset,
                            sample,
                            ])
                
            fp_heatmap = os.path.join(self.output_dir, sample, fn_heatmap)

            if overwrite or not os.path.exists(fp_heatmap):
                print(f"Plotting heatmap for directory {sensor}")
                self._plot_ig_heatmap(
                    features_unwrapped,
                    anom_ts_unwrapped,
                    gradients_features_unwrapped,
                    gradients_anom_ts_unwrapped,
                    anomaly_flag_true_unwrapped,
                    anomaly_flag_pred_class_unwrapped,
                    prediction_unwrapped,
                    sensor,
                    anomaly_date,
                    _current_output_dir,
                    _current_file_name,
                    annotate_batch_id=self.xai_config.integrated_gradients.plot.heatmap.annotate_batch_id,
                    scale_feature_gradients=self.xai_config.integrated_gradients.plot.heatmap.scale_feature_gradients,
                    dpi = self.xai_config.integrated_gradients.plot.heatmap.dpi,
                )
            else:
                print(f"File {fp_heatmap} already exists. Skipping.")
