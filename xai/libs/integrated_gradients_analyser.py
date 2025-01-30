import os
import omegaconf
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import imageio
import sys
import omegaconf as oc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.colors as mcolors
from libs.integrated_gradients import COLOR_MAP
import pandas as pd

sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")
from libs.integrated_gradients import COLOR_MAP, COLOR_NAMES
from libs.integrated_gradients import IntegratedGradientsExplainer as IG

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


CONFUSION_MATRIX = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
COLOR_TL = [(1 / 255, 183 / 255, 1), (0, 117 / 255, 177 / 255)]
ALPHA = 0.2
#green = (197 / 255, 255 / 255, 197 / 255)
green = 'green'
#orange = (255 / 255, 234 / 255, 197 / 255)
orange = 'orange'
#white = (1, 1, 1)
white = 'white'
#red = (1, 197 / 255, 197 / 255)
red = 'red'


class IntegrateGradientsAnalyser:
    def __init__(self, fp_preproc_config, fp_model_config, fp_xai_config):
        # Load configuration files
        # ----------------------------------------------------------------------
        self.fp_preproc_config = fp_preproc_config
        self.fp_model_config = fp_model_config
        self.fp_xai_config = fp_xai_config
        self.preproc_config = oc.OmegaConf.load(self.fp_preproc_config)
        self.model_config = oc.OmegaConf.load(self.fp_model_config)
        self.xai_config = oc.OmegaConf.load(self.fp_xai_config)

        # TODO
        # - USE PATH CONSTRUCTOR FROM GRADIENTEXPLAINER
        self.dp_results_parent = os.path.join(
            self.xai_config.output_dir,
            "integrated_gradients",
            self.xai_config.project,
            self.preproc_config.ds_type,
            self.xai_config.integrated_gradients.dataset,
        )

        self.stem = (
            self.xai_config.project
            + "_"
            + self.preproc_config.ds_type
            + "_"
            + self.xai_config.integrated_gradients.dataset
            + "_"
        )
        self.stem = (
            self.xai_config.project
            + "_"
            + self.preproc_config.ds_type
            + "_"
            + self.xai_config.integrated_gradients.dataset
            + "_"
        )
        self.stem_fn_1 = "anomalous_ts_" + self.stem
        self.stem_fn_2 = "ig_heatmap_" + self.stem
        self.stem_fn_concat_image = "heatmap_anomalous_ts_concat_" + self.stem
        self.conf_matrix_string = "_".join(
            [
                str(x)
                for x in sorted(
                    self.xai_config.integrated_gradients.analyser.which_samples
                )
            ]
        )

        # file name stem for predictions
        # ----------------------------------------------------------------------
        self.stem_predictions = "predictions_unwrapped"

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

    @staticmethod
    def concatenate_images_vertically(
        output_path: str,
        *image_paths: str,
        scale: float = 1,
    ) -> None:
        if not image_paths:
            raise ValueError("At least one image path must be provided.")

        # Open the first image to determine the final width
        first_img = Image.open(image_paths[0])
        common_width = int(first_img.width * scale)

        # Initialize total height
        total_height = 0

        # Iterate through image paths, open each image, resize, and calculate total height
        images = []
        for image_path in image_paths:
            try:
                img = Image.open(image_path)
            except FileNotFoundError:
                print(f"Warning: Unable to open file {image_path}")
                continue
            img = img.resize((common_width, int(img.height * scale)))
            total_height += img.height
            images.append(img)

        # Create a new image with the same width and combined height
        new_img = Image.new("RGB", (common_width, total_height), (255, 255, 255))

        # Paste the images vertically
        current_height = 0
        for img in images:
            new_img.paste(img, (0, current_height))
            current_height += img.height

        # Set the DPI of the resulting image
        new_img.save(output_path)

    @staticmethod
    def progressbar(frame: Image, frameid: int, file_paths: list, COLOR_MAP: dict):
        current_file = file_paths[frameid]
        # get the width of the image
        width = frame.size[0]
        # get the width of one rectangle
        rect_width = width / len(file_paths)
        # create 1 rectangle for each file_path
        rects = []
        for i, file_path in enumerate(file_paths):
            # get the height of the image
            height = frame.size[1]
            # get the height of the rectangle
            rect_height = height / 90
            # get the x position of the rectangle
            x = i * rect_width
            # get the y position of the rectangle
            y = height - rect_height
            # if the current file_path is the same as the file_path of the current frame
            if file_path == current_file:
                # increase width and height of the rectangle
                current_rect_width = rect_width * 5
                current_rect_height = rect_height * 2
                # create the rectangle
                rect = (x, y - current_rect_height, x + current_rect_width, y)
            else:
                # create the rectangle
                rect = (x, y, x + rect_width, y + rect_height)
            # set color based on file_path and COLOR_MAP
            # extract the true and pred values from the file_path
            true, pred = IntegrateGradientsAnalyser.get_true_pred(
                os.path.dirname(file_path)
            )
            # get the color from the COLOR_MAP
            color = COLOR_MAP[(true, pred)]
            # append the color to the rectangle
            rects.append((rect, color))
        return rects

    @staticmethod
    def create_gif(
        file_paths,
        output_path="output.gif",
        duration=500,
        loop=1,
        COLOR_MAP: dict = None,
    ):
        """
        Create a GIF from a list of file paths with an overlay showing the current image count.

        Parameters:
        - file_paths (list): List of file paths for the images in the desired order.
        - output_path (str): File path for the output GIF. Default is 'output.gif'.
        - duration (int): Time delay between each frame in milliseconds. Default is 500 milliseconds.
        - loop (int): Number of times the GIF should loop. Default is 1.
        """
        images = []

        # Read each image and append it to the images list
        for i, file_path in enumerate(file_paths):
            try:
                img = Image.open(file_path)
                draw = ImageDraw.Draw(img)

                # Font settings for the overlay
                font = ImageFont.load_default()
                font_color = (0, 0, 0)  # black
                position = (10, 10)  # upper left corner

                # Create the overlay text
                overlay_text = f"{i+1}/{len(file_paths)}"

                # Draw the overlay text on the image
                draw.text(position, overlay_text, font=font, fill=font_color)

                # draw rectangles
                rects = IntegrateGradientsAnalyser.progressbar(
                    img, i, file_paths, COLOR_MAP
                )
                for rect, color in rects:
                    draw.rectangle(rect, fill=color)

                images.append(img)
            except Exception as e:
                print(f"Warning: Unable to open file {file_path}: {e}")

        # Save as GIF
        try:
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=loop,
            )
            print(f"GIF created successfully at {output_path}")
        except Exception as e:
            print(f"Error creating GIF: {e}")

    @staticmethod
    def create_video(
        file_paths, output_path="output.mp4", fps=10, COLOR_MAP: dict = None
    ):
        """
        Create a video from a list of file paths with an overlay showing the current image count.

        Parameters:
        - file_paths (list): List of file paths for the images in the desired order.
        - output_path (str): File path for the output video. Default is 'output.mp4'.
        - fps (int): Frames per second. Default is 10.
        """
        # Create a VideoWriter object using imageio
        video_writer = imageio.get_writer(output_path, fps=fps)

        ref_width_index = None
        # Read each image and append it to the images list
        for i, file_path in enumerate(file_paths):
            print(f"\rProcessing frame {i+1}/{len(file_paths)} ", end="")
            try:
                img = Image.open(file_path)
                if not ref_width_index:
                    ref_width_index = i
            except Exception as e:
                print(f"Warning: Unable to open file {file_path}: {e}")
                continue

            draw = ImageDraw.Draw(img)

            # get the width of the first image
            if i == ref_width_index:
                width = img.size[0]
                height = img.size[1]

            # Font settings for the overlay
            font = ImageFont.load_default()
            font_color = (0, 0, 0)  # black
            position = (10, 10)  # upper left corner

            # Create the overlay text
            overlay_text = f"{i + 1}/{len(file_paths)}"

            # Draw the overlay text on the image
            draw.text(position, overlay_text, font=font, fill=font_color)

            # draw rectangles
            rects = IntegrateGradientsAnalyser.progressbar(
                img, i, file_paths, COLOR_MAP
            )
            for rect, color in rects:
                draw.rectangle(rect, fill=color)

            # Resize the image to the desired width and height after drawing
            img = img.resize((width, height))

            # Convert PIL image to NumPy array
            img_array = np.array(img)

            # Append the image array to the video writer
            video_writer.append_data(img_array)

        # Close the video writer
        video_writer.close()
        print(f"Video created successfully at {output_path}")

    @staticmethod
    def get_sensor_id(dp):
        # get the part of the string until the 4th underscore
        return "_".join(dp.split("_")[:4])

    @staticmethod
    def get_datetime(dp):
        # get the part of the string until the 4th underscore
        date_time = dp.split("_")[4:6]
        date_time = "_".join(date_time)
        return datetime.strptime(date_time, "%Y%m%d_%H%M%S")

    @staticmethod
    def get_true_pred(dp):
        # get the part of the string until the 4th underscore
        true = int(dp.split("_")[-2])
        pred = int(dp.split("_")[-1])
        return true, pred

    @staticmethod
    def remove_unwanted_directories(dns_sensors):
        unwanted_dirs = ["log", "DS_Store"]
        unwanted_files = [".png", ".npy"]

        dns_sensors = [
            dp
            for dp in dns_sensors
            if dp not in unwanted_dirs
            and not dp.startswith(".")
            and not dp.endswith(tuple(unwanted_files))
        ]

        return dns_sensors

    def get_overview(self):
        self.dns_sensors = sorted(os.listdir(self.dp_results_parent))

        # apply get_sensor_id to all elements of dns_sensors
        self.dns_sensors = [self.get_sensor_id(dp) for dp in self.dns_sensors]
        # keep only unique values
        self.dns_sensors = list(set(self.dns_sensors))
        print("Found", len(self.dns_sensors), "unique sensor IDs")

        # Usage example:
        self.dns_sensors = self.remove_unwanted_directories(self.dns_sensors)

        # # create a new output directory for the analysis
        # # ----------------------------------------------
        # create the directory
        self.output_dir_analysis = os.path.join(
            self.xai_config.output_dir,
            "integrated_gradients",
            self.xai_config.project,
            self.preproc_config.ds_type,
            self.xai_config.integrated_gradients.dataset
            + "_analysis_"
            + str(self.xai_config.integrated_gradients.analyser.concat_images_scale)
            + "_"
            + str(self.xai_config.integrated_gradients.analyser.video.video_fps),
        )
        os.makedirs(self.output_dir_analysis, exist_ok=True)

        # the directory names are structured as follows:
        # Full name: BY4168_2_BY4036_4_20190720_060000_0_0
        # sensor ID : BY4168_2_BY4036_4
        # Date: 20190720
        # Time: 060000
        # Model vs Prediction: 0_0

        # Assuming you have an initial DataFrame
        data = []
        for dn_sensors in self.dns_sensors:
            dn_samples = os.listdir(os.path.join(self.dp_results_parent, dn_sensors))
            dn_samples = self.remove_unwanted_directories(dn_samples)
            for dp in dn_samples:
                sensor_id = dn_sensors
                date_time = self.get_datetime(dp)
                true, pred = self.get_true_pred(dp)
                # create a new row as a dictionary
                new_row = {
                    "sensor_id": sensor_id,
                    "sample_name": dp,
                    "date_time": date_time,
                    "true": true,
                    "pred": pred,
                    "path": os.path.join(self.dp_results_parent, dn_sensors, dp),
                }
                data.append(new_row)

        # create the DataFrame from the list of dictionaries with specified dtype
        self.df_unfiltered = pd.DataFrame(data)
        self.df = self.df_unfiltered.copy()

        if self.xai_config.integrated_gradients.analyser.which_sensors == "all":
            self.selected_sensors = self.df["sensor_id"].unique()
        else:
            self.selected_sensors = (
                self.xai_config.integrated_gradients.analyser.which_sensors
            )
            self.df = self.df[self.df["sensor_id"].isin(self.selected_sensors)]

        if self.workerid is not None:
            print("Detectes array job. Splitting work based on sensors.")
            self.selected_sensors = IG._split_work(
                self.selected_sensors, self.workerid, self.n_worker
            )
            print("Selected sensors:", self.selected_sensors)

        # scatter plot of the results
        # ---------------------------
        # Create a new column 'color' based on 'true' and 'pred' values
        self.df["color"] = self.df[["true", "pred"]].apply(
            lambda x: COLOR_MAP[tuple(x)], axis=1
        )

        # sort df by date_time
        self.df = self.df.sort_values(by="date_time")
        # reindex
        self.df = self.df.reset_index(drop=True)

        # Make plot for time range of samples
        # ------------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 8))

        # Loop through unique sensor_ids
        for sensor_id in self.df["sensor_id"].unique():
            # Filter data for the current sensor_id
            subset = self.df[self.df["sensor_id"] == sensor_id]

            # plot a dot for each date_time value
            ax.scatter(
                subset["date_time"],
                [sensor_id] * len(subset),
                c=subset["color"],
                s=5,
                alpha=0.8,
                marker="|",
            )

        # Set labels and legend
        ax.set_xlabel("Date and Time")
        ax.set_ylabel("sensor ID")
        ax.set_title("Analysed Samples per sensor ID")
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color=color, label=label)
                for label, color in zip(COLOR_NAMES.values(), COLOR_MAP.values())
            ]
        )

        # save the plot
        plt.savefig(
            os.path.join(self.output_dir_analysis, "sensor_samples_time_range.png"),
            bbox_inches="tight",
        )

        # calculate some statistics
        # -----------------------
        # get the number of analysed samples per sensor_id
        self.df.groupby("sensor_id").count()
        # print the result only with one column
        print("Analysed samples per sensor_id:")
        print(self.df.groupby("sensor_id").count()["date_time"])
        # create new column called confusion_matrix based on COLOR_NAMES dict and the true and pred columns
        self.df["confusion_matrix"] = self.df[["true", "pred"]].apply(
            lambda x: COLOR_NAMES[tuple(x)], axis=1
        )
        # create now column for abbreviations of confusion_matrix
        self.df["confusion_matrix_abbr"] = self.df["confusion_matrix"].apply(
            lambda x: "".join([x[0] for x in x.split()])
        )

        # print counts of confusion_matrix for every sensor_id
        print("Confusion Matrix per sensor_id:")
        print(self.df.groupby("sensor_id")["confusion_matrix"].value_counts())

        # dispay the confusion matrix as horizontal bar plot
        # ------------------------------------------------------------------------------
        # get unique values of confusion_matrix which are present in self.df
        confusion_matrix_values = self.df["confusion_matrix"].unique()
        # Group by 'sensor_id' and count the occurrences of 'confusion_matrix'
        counts = (
            self.df.groupby("sensor_id")["confusion_matrix"]
            .value_counts()
            .unstack()[confusion_matrix_values]
        )
        # Plot the horizontal bar chart with stacked bars and custom colors
        ax = counts.plot(kind="barh", stacked=True, color=COLOR_MAP.values(), alpha=0.2)
        # Set legend title to none
        ax.get_legend().set_title(None)
        # save the plot
        plt.savefig(
            os.path.join(
                self.output_dir_analysis, "sensor_samples_confusion_matrix.png"
            ),
            bbox_inches="tight",
        )
        plt.close()

        print(
            "Subset the samples for selected confusion matrix values and adding surrounding samples"
        )
        self.mask = self.df[["true", "pred"]].apply(
            lambda x: CONFUSION_MATRIX[(x[0], x[1])]
            in self.xai_config.integrated_gradients.analyser.which_samples,
            axis=1,
        )

        # find all occurences in mask where the value is True and turn surr entries before and after to True
        surr = self.xai_config.integrated_gradients.analyser.keep_surrounding
        self.new_mask = self.mask.copy()
        for i, m in enumerate(self.mask):
            if m:
                self.new_mask[i - surr : i + surr + 1] = True
        self.df = self.df[self.new_mask]


        if len(self.df) == 0:
            print(
                "NoSamplesError: No samples found for the specified confusion matrix values"
            )

    def spatial_aggregate_gradients(self):
        # aggregate the gradients for each sensor along the time series
        # ------------------------------------------------------------
        stem_gradient_features_unwrapped = "gradients_features_unwrapped"
        stem_gradient_anom_ts = "gradients_anom_ts_unwrapped"

        confusion_matrix_existing = []
        for sensor_id in self.selected_sensors:
            print("Aggregating gradients for", sensor_id)
            # get all file paths of feature gradients for the current sensor_id
            fps_gradients_features = []
            fps_gradients_anom_ts = []
            # iterate over samples
            # subset the dataframe based on the sensor_id
            sample_names_current_sensor = self.df[self.df["sensor_id"] == sensor_id][
                "sample_name"
            ]
            # check if samples_names_current_sensor is empty
            if len(sample_names_current_sensor) == 0:
                print("No samples found for", sensor_id)
                continue
            else:
                for dn_sample in sample_names_current_sensor:
                    # check if sensor in dn
                    if self.get_sensor_id(dn_sample) == sensor_id:
                        # check if sample shoudl be taken
                        true, pred = self.get_true_pred(dn_sample)
                        # get confusioin matrix string for the current sample
                        confusion_matrix = CONFUSION_MATRIX[(true, pred)]
                        # check if entry in which_samples from xai_config
                        if (
                            confusion_matrix
                            in self.xai_config.integrated_gradients.analyser.which_samples
                        ):
                            confusion_matrix_existing.append(confusion_matrix)
                            # get the file path for the feature gradients
                            # print('Adding', dn_sample, 'to the list of file paths')
                            file_name_gradients_features = (
                                stem_gradient_features_unwrapped
                                + "_"
                                + self.xai_config.project
                                + "_"
                                + self.preproc_config.ds_type
                                + "_"
                                + self.xai_config.integrated_gradients.dataset
                                + "_"
                                + dn_sample
                                + ".npy"
                            )
                            file_name_gradients_anom_ts = (
                                stem_gradient_anom_ts
                                + "_"
                                + self.xai_config.project
                                + "_"
                                + self.preproc_config.ds_type
                                + "_"
                                + self.xai_config.integrated_gradients.dataset
                                + "_"
                                + dn_sample
                                + ".npy"
                            )
                            fps_gradients_features.append(
                                os.path.join(
                                    self.dp_results_parent,
                                    sensor_id,
                                    dn_sample,
                                    file_name_gradients_features,
                                )
                            )
                            fps_gradients_anom_ts.append(
                                os.path.join(
                                    self.dp_results_parent,
                                    sensor_id,
                                    dn_sample,
                                    file_name_gradients_anom_ts,
                                )
                            )
            confusion_matrix_existing = list(set(confusion_matrix_existing))
            print(
                "Found",
                len(fps_gradients_features),
                "files for feature gradients and",
                len(fps_gradients_anom_ts),
                "files for anomalous time series gradients",
            )
            # create a folder in the output_dir for the current sensor_id
            output_dir_sensor_id = os.path.join(self.output_dir_analysis, sensor_id)
            os.makedirs(output_dir_sensor_id, exist_ok=True)

            # iterate over all fps_gradients_features and concatenate the gradients
            # calculate the sum over all neighbors since the number of neigbors
            # can change for every sample
            # get the first file
            # create an empty array with the same shape as the first file summed over neighbors
            try:
                summed_features = np.zeros_like(
                    np.sum(np.load(fps_gradients_features[0]), axis=0)
                )
                summed_anom_ts = np.zeros_like(np.load(fps_gradients_anom_ts[0]))
            except:
                print("Could not load file for", sensor_id, "and", dn_sample)
                continue

            print("Aggregating feature and anomalous ts gradients...")
            for i, (fp_feature, fp_anom_ts) in enumerate(
                zip(fps_gradients_features, fps_gradients_anom_ts)
            ):
                # Calculate percentage completion
                percentage_complete = int(i / len(fps_gradients_features) * 100)
                # Print progress every 10%
                if percentage_complete % 10 == 0:
                    print(f"Progress: {percentage_complete}%")
                ##

                new_file_features = np.load(fp_feature)
                # stack the neighbors on top of each other
                new_file_features_concat = np.concatenate(new_file_features, axis=1)
                new_file_anom_ts = np.load(fp_anom_ts)

                # normalize the arrays if specified in the xai_config
                if (
                    self.xai_config.integrated_gradients.analyser.spatial_aggregation.normalize
                ):
                    combined_array = np.concatenate(
                        (new_file_features_concat, new_file_anom_ts), axis=1
                    )
                    new_file_features = new_file_features / np.max(
                        np.abs(combined_array)
                    )
                    new_file_anom_ts = new_file_anom_ts / np.max(np.abs(combined_array))

                # reduce the features (neighbors) and sum them up, divide by number of neighbors
                new_file_features = (
                    np.sum(new_file_features, axis=0) / new_file_features.shape[0]
                )

                summed_features += new_file_features
                summed_anom_ts += new_file_anom_ts

            print("Final file shape features:", summed_features.shape)
            print("Final file shape anomalous ts:", summed_anom_ts.shape)

            # devide the summed features by the number of samples
            summed_features /= len(fps_gradients_features)
            summed_anom_ts /= len(fps_gradients_anom_ts)

            # save the the aggregated gradients
            np.save(
                os.path.join(
                    output_dir_sensor_id,
                    "spatial_aggregated_gradients_features_"
                    + self.conf_matrix_string
                    + ".npy",
                ),
                summed_features,
            )
            np.save(
                os.path.join(
                    output_dir_sensor_id,
                    "spatial_aggregated_gradients_anom_ts_"
                    + self.conf_matrix_string
                    + ".npy",
                ),
                summed_anom_ts,
            )

    def concatenate_images(self):
        for i, dn_sample, sensor_id in zip(
            range(len(self.df["sample_name"])),
            self.df["sample_name"],
            self.df["sensor_id"],
        ):
            # progress bar with asci
            print(
                f"\rProcessing sample {i+1}/{len(self.df['sample_name'])} ({dn_sample})",
                end="",
            )
            fn_image1 = self.stem_fn_1 + dn_sample + ".png"
            fn_image2 = self.stem_fn_2 + dn_sample + ".png"
            fn_concat_image = self.stem_fn_concat_image + dn_sample + ".png"
            image_path1 = os.path.join(
                self.dp_results_parent, sensor_id, dn_sample, fn_image1
            )
            image_path2 = os.path.join(
                self.dp_results_parent, sensor_id, dn_sample, fn_image2
            )
            output_path = os.path.join(
                self.dp_results_parent, sensor_id, dn_sample, fn_concat_image
            )
            if (
                self.xai_config.integrated_gradients.analyser.overwrite_concat_images
                or not os.path.exists(output_path)
            ):
                self.concatenate_images_vertically(
                    output_path,
                    image_path1,
                    image_path2,
                    scale=self.xai_config.integrated_gradients.analyser.concat_images_scale,
                )
            else:
                print("Image already exists:", output_path, "Skipping...")

    def create_videos(self, sensor=None, time_from=None, time_to=None):
        # CREATE VIDEOS
        # -------------
        # select time range if specified
        if time_from is not None and time_to is not None:
            time_from = pd.to_datetime(time_from)
            time_to = pd.to_datetime(time_to)
            self.df_sub = self.df[
                (self.df["date_time"] >= time_from) & (self.df["date_time"] <= time_to)
            ]
        # iterate over all sensor_ids
        for current_sensor_id in self.selected_sensors:
            if not sensor is None and sensor != current_sensor_id:
                continue
            # progress bar with asci
            # print(f'\r{i+1}/{len(df['sensor_id'].unique())}', end='')
            # get file paths for all images of the current sensor_id
            file_paths = []
            for dn_sample in self.df_sub[self.df_sub["sensor_id"] == current_sensor_id][
                "sample_name"
            ]:
                # create a folder in the output_dir for the current sensor_id
                output_dir_sensor_id = os.path.join(
                    self.output_dir_analysis, current_sensor_id
                )
                os.makedirs(output_dir_sensor_id, exist_ok=True)
                if current_sensor_id == current_sensor_id:
                    if (
                        self.xai_config.integrated_gradients.analyser.video.videos_from_concat_images
                        == False
                    ):
                        fn_image = self.stem_fn_2 + dn_sample + ".png"
                    else:
                        fn_image = self.stem_fn_concat_image + dn_sample + ".png"
                    fp_image = os.path.join(
                        self.dp_results_parent, current_sensor_id, dn_sample, fn_image
                    )
                    file_paths.append(fp_image)
            file_paths = sorted(file_paths)
            
            print(f"Found {len(file_paths)} images for sensor_id {current_sensor_id} and time range {time_from} to {time_to}.")
            # create a video
            fps = self.xai_config.integrated_gradients.analyser.video.video_fps
            print("Creating video for", current_sensor_id)
            if time_from is not None and time_to is not None:
                output_path = os.path.join(
                    self.output_dir_analysis,
                    current_sensor_id,
                    current_sensor_id
                    + "_"
                    + time_from.strftime("%Y%m%d-%H%M%S")
                    + "_"
                    + time_to.strftime("%Y%m%d-%H%M%S")
                    + "_"
                    + str(fps)
                    + "_"
                    + self.conf_matrix_string
                    + ".mp4",
                )
            else:
                output_path = os.path.join(
                    self.output_dir_analysis,
                    current_sensor_id,
                    current_sensor_id
                    + "_"
                    + str(fps)
                    + "_"
                    + self.conf_matrix_string
                    + ".mp4",
                )

            self.create_video(
                file_paths=file_paths,
                output_path=output_path,
                fps=fps,
                COLOR_MAP=COLOR_MAP,
            )

    def plot_spatial_aggregated_gradients(self):
        which_samples = (
            self.xai_config.integrated_gradients.analyser.spatial_aggregation.which_samples
        )

        for sensor_id in self.selected_sensors:
            print("Plotting spatial aggregated gradients for", sensor_id)
            # TODO: remove log folder earlier
            if sensor_id == "log":
                continue
            output_dir_sensor_id = os.path.join(self.output_dir_analysis, sensor_id)
            n_rows = 4

            fig, ax = plt.subplots(
                len(which_samples) * n_rows, 1, figsize=(16, 8), sharex=True
            )

            vmin = -0.1
            vmax = 0.1
            ymin = 0
            ymax = 1
            colormap = "coolwarm"
            import matplotlib.colors as mcolors

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            # remove all ylabels and rotate the y label by 90 degrees
            for i in range(len(which_samples) * n_rows):
                ax[i].set_yticklabels([])
                ax[i].tick_params(axis="y", labelrotation=90)
                ax[i].set_yticklabels([])

            def plot_pcol(data, ax):
                # use the values of data_gradients as background color
                ts_length = data.shape[0]
                x = np.linspace(0, ts_length - 1, ts_length)
                y = np.linspace(ymin, ymax, ts_length)
                X, Y = np.meshgrid(x, y)
                z = data
                Z = np.tile(z, (len(z), 1))
                pcol = ax.pcolormesh(X, Y, Z, norm=norm, alpha=0.8, cmap=colormap)
                return pcol

            for i, sample in enumerate(which_samples):
                try:
                    # load the saved data
                    agg_grads_features = np.load(
                        os.path.join(
                            output_dir_sensor_id,
                            "spatial_aggregated_gradients_features_"
                            + self.conf_matrix_string
                            + ".npy",
                        )
                    )
                except:
                    print("Could not load file for", sensor_id, "and", sample)
                    continue

                agg_grads_anom_ts = np.load(
                    os.path.join(
                        output_dir_sensor_id,
                        "spatial_aggregated_gradients_anom_ts_"
                        + self.conf_matrix_string
                        + ".npy",
                    )
                )

                # print min and max of all agggregated samples
                print(
                    "Min and max of aggregated gradients for", sensor_id, "and", sample
                )
                print("Min features:", np.min(agg_grads_features))
                print("Max features:", np.max(agg_grads_features))
                print("Min anom ts:", np.min(agg_grads_anom_ts))
                print("Max anom ts:", np.max(agg_grads_anom_ts))
                

                agg_grads_features = (
                    agg_grads_features
                    * self.xai_config.integrated_gradients.analyser.spatial_aggregation.scale_feature_gradients
                )


                # plot the pcolormesh
                pcol = plot_pcol(agg_grads_anom_ts[:, 0], ax[0 + i * n_rows])
                ax[0 + i * n_rows].set_ylabel("TL1")
                pcol = plot_pcol(agg_grads_anom_ts[:, 1], ax[1 + i * n_rows])
                ax[1 + i * n_rows].set_ylabel("TL2")
                pcol = plot_pcol(agg_grads_features[:, 0], ax[2 + i * n_rows])
                ax[2 + i * n_rows].set_ylabel("TL1\nN")
                pcol = plot_pcol(agg_grads_features[:, 1], ax[3 + i * n_rows])
                ax[3 + i * n_rows].set_ylabel("TL2\nN")

                for j in range(4):
                    # add a legend on the plot just stating the sample type
                    ax[j + i * n_rows].text(
                        0.03,
                        0.5,
                        sample,
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax[j + i * n_rows].transAxes,
                        fontsize=16,
                    )

            # check if agg_grads_features exists or is not none
            if agg_grads_features is not None:
                # continue with the plot only if the data is not None

                cbar_ax = fig.add_axes(
                    [0.07, 0.2, 0.01, 0.6]
                )  # [x, y, width, height] left
                cbar = fig.colorbar(
                    pcol, cax=cbar_ax, orientation="vertical", location="left"
                )
                cbar.set_label("Attention")

                plt.text(
                    s="Spatially Averaged Attention for sensor " + sensor_id,
                    x=0.5,
                    y=0.95,
                    fontsize=16,
                    ha="center",
                    va="center",
                    transform=fig.transFigure,
                )

                # Add legend to the second plot
                # fig.tight_layout()

                # remove space between first and second row only
                fig.subplots_adjust(hspace=0)

                if (
                    self.xai_config.integrated_gradients.analyser.spatial_aggregation.normalize
                    == True
                ):
                    # save the plot in the analysis dir
                    plt.savefig(
                        os.path.join(
                            output_dir_sensor_id,
                            f"spatial_aggregated_gradients_{sensor_id}_norm.png",
                        )
                    )
                else:
                    plt.savefig(
                        os.path.join(
                            output_dir_sensor_id,
                            f"spatial_aggregated_gradients_{sensor_id}.png",
                        )
                    )
            else:
                print("No data for plotting found.")

            plt.close()

    def get_filename(self, stem, dn_sample, for_predictons=False):
        filename = (
            stem
            + "_"
            + "_".join(
                [
                    self.xai_config.project,
                    self.preproc_config.ds_type,
                    self.xai_config.integrated_gradients.dataset,
                    dn_sample,
                ]
            )
        )

        if for_predictons:
            filename = "predictons_unwrapped" + "_".join(
                [
                    self.xai_config.project,
                    self.preproc_config.ds_type,
                    self.xai_config.integrated_gradients.dataset,
                    dn_sample,
                ]
            )

        return filename

    def _scale_gradients_with_input(self):
        """
        Scales the gradients for each sample based on the selected samples
        via xai_config. Loads the saved gradients for the features and the
        anom ts and scales it with the respective input. Saves the gradients
        again and overwrites the old files.
        """

        stem_grad_features = "gradients_features_unwrapped"
        stem_grad_anom_ts = "gradients_anom_ts_unwrapped"
        stem_features = "features_unwrapped"
        stem_anom_ts = "anom_ts_unwrapped"

        # filter the dataframe based on the selected sensors
        self.df_sub = self.df[self.df["sensor_id"].isin(self.selected_sensors)]

        # remove samples from dataframe which should not be scaled
        dont_scale = pd.read_csv("../libs/config/dont_scale.txt", header=None)
        self.df_sub = self.df_sub[~self.df_sub["sample_name"].isin(dont_scale[0])]

        print(
            "! - Scaling saved gradients with respective saved data. ASSUMING ZERO BASELINE!!"
        )
        print("! - Sensors:", self.selected_sensors)

        for index in self.df_sub.index:
            # print progress for 10 percent increment
            if index % (len(self.df) / 10) == 0:
                print(f"Progress: {index}/{len(self.df)}")
            row = self.df.loc[[index]]
            try:
                grad_features = np.load(
                    os.path.join(
                        row["path"][index],
                        self.get_filename(stem_grad_features, row["sample_name"][index])
                        + ".npy",
                    )
                )
                grad_anom_ts = np.load(
                    os.path.join(
                        row["path"][index],
                        self.get_filename(stem_grad_anom_ts, row["sample_name"][index])
                        + ".npy",
                    )
                )
                features = np.load(
                    os.path.join(
                        row["path"][index],
                        self.get_filename(stem_features, row["sample_name"][index])
                        + ".npy",
                    )
                )
                anom_ts = np.load(
                    os.path.join(
                        row["path"][index],
                        self.get_filename(stem_anom_ts, row["sample_name"][index])
                        + ".npy",
                    )
                )
            except:
                print("! - Could not load file for", row["sample_name"][index])
                continue

            grad_features = grad_features * features
            grad_anom_ts = grad_anom_ts * anom_ts

            np.save(
                os.path.join(
                    row["path"][index],
                    self.get_filename(stem_grad_features, row["sample_name"][index])
                    + ".npy",
                ),
                grad_features,
            )
            np.save(
                os.path.join(
                    row["path"][index],
                    self.get_filename(stem_grad_anom_ts, row["sample_name"][index])
                    + ".npy",
                ),
                grad_anom_ts,
            )
            print("! - Scaled sample", row["sample_name"][index])

    def _rename_based_on_threshold(self, for_predictons=False):
        """
        Rename the directories and containing files based on the prediction threshold.
        """
        # filter the dataframe based on the selected sensors
        self.df_sub = self.df[self.df["sensor_id"].isin(self.selected_sensors)]

        print("Renaming files based on prediction threshold...")
        for index in self.df_sub.index:
            dn_sample = (
                self.get_filename(
                    self.stem_predictions,
                    self.df_sub.loc[index, "sample_name"],
                    for_predictons=for_predictons,
                )
                + ".npy"
            )
            dp_sample = self.df_sub.loc[index, "path"]
            file_path = os.path.join(dp_sample, dn_sample)
            try:
                prediction = np.load(file_path)
            except:
                print(
                    "! - Could not load file for", self.df_sub.loc[index, "sample_name"]
                )
                continue
            if prediction > self.xai_config.integrated_gradients.threshold:
                pred_class = 1
            else:
                pred_class = 0
            # do the same for all files in the directory
            for file in os.listdir(dp_sample):
                file_name, file_ext = os.path.splitext(file)
                new_file_name = file_name[:-1] + str(pred_class) + file_ext
                os.rename(
                    os.path.join(dp_sample, file),
                    os.path.join(dp_sample, new_file_name),
                )
            # rename the dp_sample in the os
            new_dp_sample = dp_sample[:-1] + str(pred_class)
            os.rename(dp_sample, new_dp_sample)

        self.get_overview()


    @staticmethod
    def get_similarity_idx(features_before, features):
        """
        Returns a list of tuples containing the indices of rows in `features_before` that have similar rows in `features`.

        Parameters:
        - features_before (numpy.ndarray): The array of features before the change.
        - features (numpy.ndarray): The array of features after the change.

        Returns:
        - list: A list of tuples, where each tuple contains the indices of a similar row in `features_before` and `features`.
               If a row in `features_before` does not have a similar row in `features`, the second index in the tuple is `nan`.
        """
        same_rows_idx = []
        for i in range(features_before.shape[0]):
            found = False
            for j in range(features.shape[0]):
                if np.all(np.isclose(features_before[i, 1:, :], features[j, :-1, :], rtol=0.1)):
                    same_rows_idx.append((i, j))
                    found = True
                elif j == features.shape[0] - 1 and not found:
                    same_rows_idx.append((i, np.nan))
        return same_rows_idx


    @staticmethod
    def construct_new_array(features_before, features, similarity_idx):
        """
        Constructs a new array by replacing certain elements in the original array with elements from another array.

        Parameters:
        - features_before (ndarray): The original array of features before replacement.
        - features (ndarray): The array of features to be used for replacement.
        - similarity_idx (ndarray): The indices indicating which elements to replace and with which elements.

        Returns:
        - new_features (ndarray): The new array with replaced elements.

        """
        new_features = np.zeros(features_before.shape)
        for idx in similarity_idx:
            if np.isnan(idx[0]) or np.isnan(idx[1]):
                new_features[idx[0], :, :] = np.nan
            else:
                new_features[idx[0], :, :] = features[idx[1], :, :]
        return new_features


    def plot_agg_samples_over_time(self, sensor, time_from=None, time_to=None, agg_type=None, norm_by_prediction=None, cbar_limits=None):
        stem_anom_ts_unwrapped = "anom_ts_unwrapped"
        stem_anomaly_flag_true_unwrapped = "anomaly_flag_true_unwrapped"
        stem_features_unwrapped = "features_unwrapped"
        stem_gradients_anom_ts_unwrapped = "gradients_anom_ts_unwrapped"
        stem_gradients_features_unwrapped = "gradients_features_unwrapped"
        stem_predictions_unwrapped = "predictions_unwrapped"

        def _load_sample(path):
            """
            load sample values based on given path
            """

            anom_ts = np.load(
                os.path.join(
                    row["path"][index],
                    self.get_filename(stem_anom_ts_unwrapped, row["sample_name"][index])
                    + ".npy",
                )
            )
            time = self.get_datetime(row["sample_name"][index])
            prediction = np.load(
                os.path.join(
                    row["path"][index],
                    self.get_filename(
                        stem_predictions_unwrapped, row["sample_name"][index]
                    )
                    + ".npy",
                )
            )
            flag_true = np.load(
                os.path.join(
                    row["path"][index],
                    self.get_filename(
                        stem_anomaly_flag_true_unwrapped, row["sample_name"][index]
                    )
                    + ".npy",
                )
            )
            gradients_anom_ts = np.load(
                os.path.join(
                    row["path"][index],
                    self.get_filename(
                        stem_gradients_anom_ts_unwrapped, row["sample_name"][index]
                    )
                    + ".npy",
                )
            )
            gradients_features = np.load(
                os.path.join(
                    row["path"][index],
                    self.get_filename(
                        stem_gradients_features_unwrapped, row["sample_name"][index]
                    )
                    + ".npy",
                )
            )
            features = np.load(
                os.path.join(
                    row["path"][index],
                    self.get_filename(
                        stem_features_unwrapped, row["sample_name"][index]
                    )
                    + ".npy",
                )
            )

            return (
                anom_ts,
                time,
                prediction,
                flag_true,
                gradients_anom_ts,
                gradients_features,
                features,
            )

        def agg_gradients(gradients, agg_type, axis):
            # mean, sum, max, min
            if agg_type == "mean":
                return np.nanmean(gradients, axis=axis)
            elif agg_type == "sum":
                return np.nansum(gradients, axis=axis)
            elif agg_type == "max":
                return np.nanmax(gradients, axis=axis)
            elif agg_type == "min":
                return np.nanmin(gradients, axis=axis)

        if agg_type is None:
            agg_type = (
                self.xai_config.integrated_gradients.analyser.aggregate_sample_along_time.agg_type
            )
        # only execute for selected sensors
        self.df_sub = self.df_unfiltered[self.df_unfiltered["sensor_id"].isin([sensor])]
        
        if time_from and time_to is not None:
            # subset based on time range
            time_from = pd.to_datetime(
                time_from
            )
            time_to = pd.to_datetime(
                time_to
            )
            self.df_sub = self.df_sub[
                (self.df_sub["date_time"] >= time_from)
                & (self.df_sub["date_time"] <= time_to)
            ]

        interval = self.xai_config.integrated_gradients.analyser.aggregate_sample_along_time.interval
        interval = pd.Timedelta(interval, unit="s")
        # generate a list of datetime object starting with time_from and ending with time_to with interval
        
        if time_from and time_to is None:
            time_from = self.df_sub["date_time"].min()
            time_to = self.df_sub["date_time"].max()
        time_range = pd.date_range(start=time_from, end=time_to, freq=interval)

        for sensor in [sensor]:
            self.df_sub_sensor = self.df_sub[self.df_sub["sensor_id"] == sensor]

            anom_ts = []
            time = []
            predictions = []
            flag_true = []
            gradients_anom_ts = []
            gradients_features = []
            features = []

            ts_index = self.preproc_config.timestep_before

            print('Aggregating gradients...')
            for i, curr_datetime in enumerate(time_range):
                if i % 100 == 0:
                    print(i, "/", len(time_range))
                try:
                    # get the index of the current datetime
                    index = self.df_sub_sensor[
                        self.df_sub_sensor["date_time"] == curr_datetime
                    ].index[0]
                except IndexError:
                    print("No data found for the given date time", curr_datetime)
                    ts_length = self.preproc_config.timestep_before + self.preproc_config.timestep_after + 1
                    anom_ts_ = np.full((ts_length, 2), np.nan)
                    time_ = curr_datetime
                    predictions_ = [np.nan]
                    flag_true_ = np.nan
                    gradients_anom_ts_ = np.full((ts_length, 2), np.nan)
                    try:
                        gradients_features_ = np.full((gradients_features_shape0[0], ts_length, 2), np.nan)
                    except UnboundLocalError:
                        raise ValueError("Not data found for the first frame in the time series. Start with a sample where data is available.")

                    features_ = np.full((gradients_features_shape0[0], ts_length, 2), np.nan)
                else:
                    row = self.df_sub_sensor.loc[[index]]

                    # load
                    (
                        anom_ts_,
                        time_,
                        predictions_,
                        flag_true_,
                        gradients_anom_ts_,
                        gradients_features_,
                        features_,
                    ) = _load_sample(row["path"][index])


                    # if i > 0:
                    #     # compare the loaded featured with the features from the step before to ensure that the neighbors did not change from one sample to another
                          # The comparison is based on the features, i.e. the time series of the neighbors itself. If neighbors are missing in the next sample, the corresponding row in the features is filled with nans.
                          # If neighbors are added from one sample to another, then the new neighbors are discarded. The starting date (sample) will influence the final result. 
                    #     similarity_idx = self.get_similarity_idx(features_before, features_)
                    #     features_ = self.construct_new_array(features_before, features_, similarity_idx)
                    #     gradients_features_ = self.construct_new_array(gradients_features_before, gradients_features_, similarity_idx)
                    #     # count the number of nans in the features_before
                    #     n_nans = np.sum(np.isnan(features_[:, ts_index, :]))
                    #     if n_nans > 0:
                    #         print('Number of nan rows in adapted features_:', n_nans)
                    #         print('Number of nan rows in adapted gradients_geatures_:', n_nans)
                    #         print('Current datetime:', curr_datetime)

                    if norm_by_prediction is None:
                        norm_by_prediction = self.xai_config.integrated_gradients.analyser.aggregate_sample_along_time.norm_by_prediction
                        
                    if norm_by_prediction == True:
                        # normalize the gradients by the prediction
                        gradients_anom_ts_ = gradients_anom_ts_ / predictions_
                        gradients_features_ = gradients_features_ / predictions_

                    # scale the gradients_features for visualization
                    if self.xai_config.integrated_gradients.analyser.aggregate_sample_along_time.scale_feature_gradients == 'auto':
                        gradients_features_ = gradients_features_ / gradients_features_.shape[0] * 2
                    else:                    
                        gradients_features_ = (
                            gradients_features_
                            * self.xai_config.integrated_gradients.analyser.aggregate_sample_along_time.scale_feature_gradients
                        )


                    # save the shape of the first 
                    if i == 0:
                        gradients_features_shape0 = gradients_features_.shape
                    else:
                        if gradients_features_.shape != gradients_features_shape0:
                            print("Shapes of gradients_features do not match.")
                            print("Shape of current gradients_features:", gradients_features_.shape, "Shape of first gradients_features in time series:", gradients_features_shape0)
                            print('Cheking which neighbor is missing:')
                            gradients_features_ = np.full(gradients_features_shape0, np.nan)
                            features_ = np.full(gradients_features_shape0, np.nan)
                            pass


                    features_before = features_
                    gradients_features_before = gradients_features_

                anom_ts.append(anom_ts_[ts_index, :])
                time.append(time_)
                predictions.append(float(predictions_[0]))
                flag_true.append(flag_true_)
                gradients_anom_ts.append(agg_gradients(gradients_anom_ts_, agg_type, 0))
                gradients_features.append(
                    agg_gradients(gradients_features_, agg_type, 1)
                )
                features.append(features_[:, ts_index, :])

                #print('Shape of gradients_features:', np.array(gradients_features).shape)

            anom_ts = np.array(anom_ts)
            time = np.array(time)
            predictions = np.array(predictions)
            flag_true = np.array(flag_true)
            gradients_anom_ts = np.array(gradients_anom_ts)
            gradients_features = np.array(gradients_features)
            features = np.array(features)

            if cbar_limits is None:
                cbar_limits = self.xai_config.integrated_gradients.analyser.aggregate_sample_along_time.cbar_limits
            # get min max for colorbar
            if (
                not cbar_limits
                == "auto"
            ):
                vmin = cbar_limits[
                    0
                ]
                vmax = cbar_limits[
                    1
                ]
            else:
                vmin = tf.reduce_min(gradients_features)
                vmax = tf.reduce_max(gradients_features)

            if self.xai_config.integrated_gradients.analyser.aggregate_sample_along_time.group_tl_channels == False:
                rows = 1 + 2 + gradients_features.shape[1] * 2
            else:
                rows = 1 + 2 + gradients_features.shape[1]
            norm2 = mcolors.Normalize(vmin=vmin, vmax=vmax)
            
            colormap = "RdBu_r"
            # cmap_over_norm = plt.cm.get_cmap(colormap, 256).copy()
            # cmap_over_norm.set_over('lightgrey')

            threshold = self.xai_config.integrated_gradients.threshold
            flag_prediction = np.where(predictions > threshold, 1, 0)

            COLOR_VALUES = {
                (0, 0): 0,  # "white",  # True Negative, TN
                (0, 1): 0.33,  # "orange",  # False Positive, FP
                (1, 0): 0.66,  # "red",  # False Negative, FN
                (1, 1): 1,  # "green",  # True Positive, TP
            }

            colors_vals = []
            for true, pred in zip(flag_true, flag_prediction):
                # check if true or pred is none:
                if np.isnan(true) or np.isnan(pred):
                    colors_vals.append(0)
                else:
                    colors_vals.append(COLOR_VALUES[(true, pred)])
            colors_vals = np.array(colors_vals)

            colors = [white, orange, red, green]
            positions = list(COLOR_VALUES.values())
            from matplotlib.colors import LinearSegmentedColormap

            # Create the colormap
            custom_cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", list(zip(positions, colors))
            )

            # FIGURE CREATION
            # ----------------
            ts_length = len(time_range)
            X = np.linspace(-0.5, ts_length - 0.5, ts_length + 1)
            x = X[:-1] + 0.5

            height_ratios = [1] + [1] * (rows - 1)  # Adjust height ratios as needed
            gs_top = plt.GridSpec(
                rows,
                1,
                top=0.9,
                bottom=0.13,
                height_ratios=height_ratios,
            )
            gs_mid = plt.GridSpec(
                rows,
                1,
                top=0.80,
                bottom=0.13,
                height_ratios=height_ratios,
            )
            gs_base = plt.GridSpec(
                rows,
                1,
                top=0.75,
                bottom=0.13,
                height_ratios=height_ratios,
            )

            ylims = self.xai_config.integrated_gradients.analyser.aggregate_sample_along_time.ylims
            ymin = ylims[0]
            ymax = ylims[1]
            
            width = ts_length/300
            height = 2 + 0.5 + gradients_features.shape[1]
            fig = plt.figure(figsize=(width, height))

            #font_size_paper = 10  # for fist sensor
            font_size_paper = 6  # for Second sensor
            # Calculate the font size relative to the figure size
            font_size_ratio = min(width, height) / 6 
            font_size = font_size_paper * font_size_ratio

            # Set the font size globally
            plt.rcParams.update({'font.size': font_size})

            # PLOT  TOP
            # ---------
            topax = fig.add_subplot(gs_top[0])
            # plot TL
            # topax.plot(x, anom_ts[:, 0], color=COLOR_TL[0])
            # topax.set_ylabel("TL", rotation=0, labelpad=20)
            # topax.plot(x, anom_ts[:, 1], color=COLOR_TL[1])
            # topax.set_ylim(ylims[0], ylims[1])
            # topax.set_title("Flagged Sensor Time Series and Model Prediction")
            # plot prediction
            #topax_2 = topax.twinx()
            topax.plot(x, predictions, color="k")
            topax.set_ylim(0, 1)
            topax.set_ylabel("Prediction")
            # set background color according to flag_true and flag_prediction
            if ylims == "auto":
                Y = [np.min(anom_ts), np.max(anom_ts)]
            else:
                Y = [ymin, ymax]
            z = colors_vals
            Z = np.reshape(z, (1, ts_length))
            pcol1 = topax.pcolormesh(
                X, Y, Z, alpha=ALPHA, cmap=custom_cmap, vmin=0, vmax=1
            )
            plt.setp(topax.get_xticklabels(), visible=False)

            # add an artificial legend with colors from the colors list below the plot
            from matplotlib.patches import Patch
            # Create patches for the legend
            patches = [Patch(facecolor=color, label=label, edgecolor='grey', linewidth=1, alpha=ALPHA) for color, label in zip(colors, ["TN", "FP", "FN", "TP"])]
            topax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(colors), frameon=False, edgecolor='grey')
            # remove the title from the legend
            topax.get_legend().set_title(None)

            # PLOT MID
            # ---------
            for i in range(gradients_anom_ts.shape[1]):
                midax = fig.add_subplot(gs_mid[i + 1, :])
                midax.plot(x, anom_ts[:, i], color=COLOR_TL[i])
                if ylims == "auto":
                    Y = [np.min(anom_ts), np.max(anom_ts)]
                else:
                    midax.set_ylim(ylims[0], ylims[1])
                    Y = [ymin, ymax]
                z = gradients_anom_ts[:, i]
                Z = np.reshape(z, (1, ts_length))
                pcol2 = midax.pcolormesh(X, Y, Z, norm=norm2, alpha=0.8, cmap=colormap)
                # remove xticks and labels
                midax.set_xticklabels([])
                if i == 0:
                    midax.set_title("Flagged Sensor")
                    midax.set_ylabel("TL 1", rotation=0, labelpad=50, loc="bottom")
                else:
                    midax.set_ylabel("TL 2", rotation=0, labelpad=50, loc="bottom")
            
            # PLOT BASE for grouped TL1 and TL2 channels
            # ------------------------------------------
            if self.xai_config.integrated_gradients.analyser.aggregate_sample_along_time.group_tl_channels == True:
                gradients_features = np.mean(gradients_features, axis=2)

                index = 3
                for i in range(gradients_features.shape[1]):
                    baseax = fig.add_subplot(gs_base[i+index])
                    baseax.plot(x, features[:, i, 0], color=COLOR_TL[0])
                    baseax.plot(x, features[:, i, 1], color=COLOR_TL[1])
                    
                    if ylims == "auto":
                        Y = [np.min(anom_ts), np.max(anom_ts)]
                    else:
                        baseax.set_ylim(ylims[0], ylims[1])
                        Y = [ymin, ymax]
                    # use the values of data_gradients as background color
                    z = gradients_features[:, i]
                    Z = np.reshape(z, (1, ts_length))
                    pcol2 = baseax.pcolormesh(
                        X, Y, Z, norm=norm2, alpha=0.8, cmap=colormap
                    )
                    if i == 0:
                        baseax.set_title(
                            "Self Reference Cycle and Neighbouring Sensors"
                        )
                    
                    baseax.set_ylabel(
                        "N" + str(i) + " TL",
                        rotation=0,
                        labelpad=50,
                        loc="center",
                    )
                    # remove xticks and labels for i=0
                    if i != gradients_features.shape[1] - 1:
                        baseax.set_xticks([])
                        baseax.set_xticklabels([])
    

            # PLOT BASE for seperated TL1 and TL2 channels
            # ------------------------------------------
            if self.xai_config.integrated_gradients.analyser.aggregate_sample_along_time.group_tl_channels == False:
                # PLOT BASE
                # ---------
                index = 3
                for i in range(gradients_features.shape[1]):
                    for j in range(2):
                        baseax = fig.add_subplot(gs_base[index + j])
                        baseax.plot(x, features[:, i, j], color=COLOR_TL[j])
                        if ylims == "auto":
                            Y = [np.min(anom_ts), np.max(anom_ts)]
                        else:
                            baseax.set_ylim(ylims[0], ylims[1])
                            Y = [ymin, ymax]
                        # use the values of data_gradients as background color
                        z = gradients_features[:, i, j]
                        Z = np.reshape(z, (1, ts_length))
                        pcol2 = baseax.pcolormesh(
                            X, Y, Z, norm=norm2, alpha=0.8, cmap=colormap
                        )
                        if i == 0 and j == 0:
                            baseax.set_title(
                                "Self Reference Cycle and Neighbouring Sensors"
                            )
                        if j == 0:
                            baseax.set_ylabel(
                                "N" + str(i) + " TL1",
                                rotation=0,
                                labelpad=50,
                                loc="center",
                            )
                        else:
                            baseax.set_ylabel(
                                "N" + str(i) + " TL2",
                                rotation=0,
                                labelpad=50,
                                loc="center",
                            )

                        # remove xticks and labels for i=0
                        if i != gradients_features.shape[1] - 1:
                            baseax.set_xticks([])
                            baseax.set_xticklabels([])
                    index += 2

            # set xlims to keep the missing values at the end of a time series
            plt.xlim(x[0], x[-1])
            # print('Check if time is equal to time_range:')
            # print(time == time_range)
            # print('Check if len are all the same')
            # print('Length of time:', len(time))
            # print('Length of time_range:', len(time_range))
            # print('Length of x:', len(x))
            # convert time to string
            time = [str(t) for t in time]
            # remove seconds from time
            time = [t[:-3] for t in time]

            # display x ticks only for times of 8 pm
            xticks = []
            xticklabels = []
            for i, t in enumerate(time):
                #if t[-5:] == "20:00": # for first sensor
                if t[-5:] == "06:00": # for second sensor
                    xticks.append(x[i])
                    xticklabels.append(t)
            plt.xticks(xticks, xticklabels)

            # colorbar
            # Define the position and size of the colorbar
            cbar_width_percentage = (
                50  # Percentage of the width the colorbar should span
            )
            cbar_height = 0.2  # Height of the colorbar in inches
            cbar_x = (
                100 - cbar_width_percentage
            ) / 2  # X-position of the bottom-left corner of the colorbar
            #cbar_y = 0.65  # Y-position of the bottom-left corner of the colorbar # for first sensor
            cbar_y = 0.95  # Y-position of the bottom-left corner of the colorbar # for second sensor


            # Convert percentages to relative coordinates
            fig_width_inches = fig.get_size_inches()[0]
            fig_height_inches = fig.get_size_inches()[1]
            cbar_x_rel = cbar_x / 100
            cbar_y_rel = cbar_y / fig_height_inches
            cbar_width_rel = cbar_width_percentage / 100
            cbar_height_rel = cbar_height / fig_height_inches

            # Add the colorbar
            cbar_ax = fig.add_axes(
                [cbar_x_rel, cbar_y_rel, cbar_width_rel, cbar_height_rel]
            )
            cbar = fig.colorbar(pcol2, cax=cbar_ax, orientation="horizontal")
            cbar.set_label("Attribution")

            plt.tight_layout()
            if norm_by_prediction:
                plt.savefig(
                    os.path.join(
                        self.output_dir_analysis,
                        "agg_samples_over_time_" + sensor + '_' + time_from.strftime("%Y%m%d-%H%M%S") + "-" + time_to.strftime("%Y%m%d-%H%M%S") + "_" + agg_type + "_norm.png",
                    )
                )
            else:
                plt.savefig(
                    os.path.join(
                        self.output_dir_analysis,
                        "agg_samples_over_time_" + sensor + '_' + time_from.strftime("%Y%m%d-%H%M%S") + "-" + time_to.strftime("%Y%m%d-%H%M%S") + "_" + agg_type + ".png",
                    )
                )