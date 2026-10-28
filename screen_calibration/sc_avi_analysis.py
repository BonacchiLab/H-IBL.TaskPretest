"""Analyze AVI videos for screen calibration, focusing on RGB intensity,
luminance, and spatial uniformity.
It is directly compatible with the bonsai workflow for screen calibration, but can also be run as a standalone script.
It analyzes RGB intensity, luminance, spatial uniformity, and per-frame
RGB/luminance changes in screen-calibration AVI videos. It functions for the:
- Control screens (black-white-black-blue-black-red-black-green-black)
- Grey sweeps
- Blue sweeps
- Red sweeps
- Green sweeps

Videos recorded by Bonsai can contain the full sequence. When a matching
timestamps_sc*.csv file is present, each AVI is split into screen states using
the logged RGB values and timestamps. Videos without a matching timing CSV are
analyzed as single screens. Files may be placed directly in the data directory
or in subfolders named PC_01, PC_02, and so on. Statistical display comparisons
use independent AVI recordings within each PC_* folder; individual video frames
are not treated as independent samples.
"""

import argparse
import glob
from itertools import combinations
import os
import re
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# This function is used to calculate the mean RGB values for a ROI. 
def _roi_channel_means(rgb_frame, roi_box): 
    """Return RGB means for a normalized (x_min, y_min, x_max, y_max) ROI."""
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        raise ValueError("Expected a video frame with three RGB channels.")

    height, width = rgb_frame.shape[:2]
    x_min = int(width * roi_box[0])
    y_min = int(height * roi_box[1])
    x_max = int(width * roi_box[2])
    y_max = int(height * roi_box[3])
    roi = rgb_frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        raise ValueError(f"ROI {roi_box} is empty for a {width}x{height} frame.")

    return roi.mean(axis=(0, 1))

# This function analyzes a video file and calculates the mean RGB values for the center and corner regions of each frame.
# It also computes the luminance and uniformity percentage based on these values.
def analyze_video(file_path):
    """Measure mean center and top-left RGB intensity across all video frames."""
    capture = cv2.VideoCapture(file_path)
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open AVI video: {file_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    center_sum = np.zeros(3, dtype=np.float64)
    corner_sum = np.zeros(3, dtype=np.float64)

    try:
        while True:
            success, bgr_frame = capture.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            center_sum += _roi_channel_means(rgb_frame, (0.4, 0.4, 0.6, 0.6))
            corner_sum += _roi_channel_means(rgb_frame, (0.05, 0.05, 0.15, 0.15))
            frame_count += 1
    finally:
        capture.release()

    if frame_count == 0: #debug 
        raise ValueError(f"No readable frames found in AVI video: {file_path}")

    center_rgb = center_sum / frame_count
    corner_rgb = corner_sum / frame_count
    center_luminance = float(np.dot(center_rgb, (0.2126, 0.7152, 0.0722)))
    corner_luminance = float(np.dot(corner_rgb, (0.2126, 0.7152, 0.0722)))

    return {
        "Red_Center": center_rgb[0],
        "Green_Center": center_rgb[1],
        "Blue_Center": center_rgb[2],
        "Luminance_Center": center_luminance,
        "Uniformity_Corner_pct": (
            corner_luminance / center_luminance * 100
            if center_luminance > 0
            else 0
        ),
        "Frame_Count": frame_count,
        "FPS": fps if np.isfinite(fps) and fps > 0 else np.nan,
        "Duration_Seconds": (
            frame_count / fps if np.isfinite(fps) and fps > 0 else np.nan
        ),
    }

# Function for csv timestamp extraction from file name
def _timestamp_token(file_name): 
    """Return the Bonsai timestamp shared by an AVI and its timing CSV."""
    match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}", file_name)
    return match.group(0) if match else None

# This function converts RGB values to a readable screen name based on the intensity and active channels.
def _screen_name(rgb):
    """Convert a Bonsai RGB stimulus value to a readable screen name."""
    channels = np.asarray(rgb, dtype=np.float64)
    channel_names = ("Red", "Green", "Blue")

    if np.allclose(channels, channels[0], atol=1e-4):
        intensity = int(round(channels[0] * 100))
        if intensity == 0:
            return "BlackScreen"
        if intensity == 100:
            return "WhiteScreen"
        return f"Grey{intensity}Screen"

    active_channels = np.flatnonzero(np.abs(channels) > 1e-4)
    if len(active_channels) == 1:
        channel = active_channels[0]
        intensity = int(round(channels[channel] * 100))
        name = channel_names[channel]
        if intensity == 100:
            return f"{name}Screen"
        return f"{name}{intensity}Screen"

    return "RGB_" + "_".join(f"{value:g}" for value in channels)

# Function to load screen segments from a CSV file and map them onto AVI frame ranges.
def _load_screen_segments(csv_path, frame_count):
    """Map Bonsai's per-frame RGB values and timestamps onto AVI frame ranges."""
    timing = pd.read_csv(csv_path)
    required_columns = {"Value.X", "Value.Y", "Value.Z", "Timestamp"}
    missing_columns = required_columns.difference(timing.columns)
    if missing_columns:
        raise ValueError(
            f"Timing CSV is missing required columns {sorted(missing_columns)}: "
            f"{csv_path}"
        )
    if timing.empty:
        raise ValueError(f"Timing CSV contains no screen data: {csv_path}")

    colors = timing[["Value.X", "Value.Y", "Value.Z"]].to_numpy(dtype=np.float64)
    timestamps = pd.to_datetime(timing["Timestamp"], errors="coerce", utc=True)
    if not np.isfinite(colors).all() or timestamps.isna().any():
        raise ValueError(f"Timing CSV contains invalid screen data: {csv_path}") 

    timestamp_seconds = timestamps.astype("int64").to_numpy() / 1e9
    intervals = np.diff(timestamp_seconds)
    positive_intervals = intervals[intervals > 0]
    if not len(positive_intervals):
        if len(timestamp_seconds) > 1:
            raise ValueError(f"Timing CSV timestamps are not increasing: {csv_path}") 
        median_interval = 1.0
    else:
        if np.any(intervals < 0):
            raise ValueError(f"Timing CSV timestamps are not increasing: {csv_path}")
        median_interval = float(np.median(positive_intervals))

    total_duration = timestamp_seconds[-1] - timestamp_seconds[0] + median_interval
    if total_duration <= 0:
        raise ValueError(f"Timing CSV has an invalid duration: {csv_path}")

    run_starts = np.r_[0, np.flatnonzero(np.any(np.diff(colors, axis=0) != 0, axis=1)) + 1]
    segments = [] 
    for index, row_start in enumerate(run_starts):
        start_time = timestamp_seconds[row_start] - timestamp_seconds[0]
        start_frame = int(round(start_time / total_duration * frame_count))
        if index + 1 < len(run_starts):
            end_time = timestamp_seconds[run_starts[index + 1]] - timestamp_seconds[0]
            end_frame = int(round(end_time / total_duration * frame_count))
        else:
            end_frame = frame_count

        if end_frame <= start_frame:
            raise ValueError(
                f"Timing CSV defines an empty screen segment at step {index + 1}: "
                f"{csv_path}"
            )
        segments.append(
            (start_frame, end_frame, _screen_name(colors[row_start]), index + 1)
        )

    return segments

# Function to find video folders in the data directory, returning either the PC_* subfolders or the directory itself 
# if none are found.
def _video_folders(data_directory):
    """Return the PC_* data folders, or the directory itself if there are none."""
    pc_folders = sorted(
        (
            folder
            for folder in glob.glob(os.path.join(data_directory, "PC_*"))
            if os.path.isdir(folder)
        ),
        key=str.lower,
    )
    return pc_folders or [data_directory]

# Function to find AVI files in a given folder, returning a sorted list of file paths.
def _avi_files(folder):
    """Return AVI files directly inside a data folder."""
    return sorted(
        (
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith(".avi")
            and os.path.isfile(os.path.join(folder, name))
        ),
        key=str.lower,
    )

# This function analyzes a video file and its corresponding timing CSV to measure each changing Bonsai screen state 
# in a continuous AVI recording.
def analyze_video_sweeps(file_path, timing_csv):
    """Measure each changing Bonsai screen state in a continuous AVI recording."""
    capture = cv2.VideoCapture(file_path)
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open AVI video: {file_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    center_means = []
    corner_means = []
    try:
        while True:
            success, bgr_frame = capture.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            center_means.append(
                _roi_channel_means(rgb_frame, (0.4, 0.4, 0.6, 0.6))
            )
            corner_means.append(
                _roi_channel_means(rgb_frame, (0.05, 0.05, 0.15, 0.15))
            )
    finally:
        capture.release()

    frame_count = len(center_means)
    if frame_count == 0:
        raise ValueError(f"No readable frames found in AVI video: {file_path}")

    center_means = np.asarray(center_means)
    corner_means = np.asarray(corner_means)
    results = []
    for start_frame, end_frame, screen_name, step in _load_screen_segments(
        timing_csv, frame_count
    ):
        center_rgb = center_means[start_frame:end_frame].mean(axis=0)
        corner_rgb = corner_means[start_frame:end_frame].mean(axis=0)
        center_luminance = float(
            np.dot(center_rgb, (0.2126, 0.7152, 0.0722))
        )
        corner_luminance = float(
            np.dot(corner_rgb, (0.2126, 0.7152, 0.0722))
        )
        results.append(
            {
                "Step": step,
                "Screen": screen_name,
                "Red_Center": center_rgb[0],
                "Green_Center": center_rgb[1],
                "Blue_Center": center_rgb[2],
                "Luminance_Center": center_luminance,
                "Uniformity_Corner_pct": (
                    corner_luminance / center_luminance * 100
                    if center_luminance > 0
                    else 0
                ),
                "Frame_Count": end_frame - start_frame,
                "FPS": fps if np.isfinite(fps) and fps > 0 else np.nan,
                "Duration_Seconds": (
                    (end_frame - start_frame) / fps
                    if np.isfinite(fps) and fps > 0
                    else np.nan
                ),
            }
        )

    return results

# Compares AVI videos in PC_* subfolders or directly in the given directory, returning a DataFrame with the results.
def analyze_all_computers(data_directory):
    """Analyze AVI videos in PC_* subfolders or directly in the given directory."""
    if not os.path.isdir(data_directory):
        raise FileNotFoundError(f"Data directory does not exist: {data_directory}")

    results = []
    for folder in _video_folders(data_directory):
        timing_files = {}
        for name in os.listdir(folder):
            if name.lower().endswith(".csv"):
                token = _timestamp_token(name)
                if token:
                    if token in timing_files:
                        raise ValueError(
                            f"Multiple timing CSV files share timestamp {token} "
                            f"in {folder}."
                        )
                    timing_files[token] = os.path.join(folder, name)

        pc_id = (
            os.path.basename(folder)
            if os.path.normcase(folder) != os.path.normcase(data_directory)
            else "Single_Folder"
        )
        for file_path in _avi_files(folder):
            token = _timestamp_token(os.path.basename(file_path))
            timing_csv = timing_files.get(token) if token else None
            recording = token or os.path.splitext(os.path.basename(file_path))[0]
            if timing_csv:
                results.extend(
                    {
                        "PC_ID": pc_id,
                        "Recording": recording,
                        **result,
                    }
                    for result in analyze_video_sweeps(file_path, timing_csv)
                )
                continue

            results.append(
                {
                    "PC_ID": pc_id,
                    "Recording": recording,
                    "Screen": os.path.splitext(os.path.basename(file_path))[0],
                    **analyze_video(file_path),
                }
            )

    return pd.DataFrame(results)

#Plots a time series for each RGB channel with one curve per AVI.
def plot_video_time_series(data_directory, show=True):
    """Plot a time series for each RGB channel with one curve per AVI."""
    video_files = [
        file_path
        for folder in _video_folders(data_directory)
        for file_path in _avi_files(folder)
    ]
    if not video_files:
        print("No AVI video data found for time-series plotting.")
        return

    video_series = []
    for file_path in video_files:
        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Could not open AVI video: {file_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        center_means = []
        try:
            while True:
                success, bgr_frame = capture.read()
                if not success:
                    break

                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                center_means.append(
                    _roi_channel_means(rgb_frame, (0.4, 0.4, 0.6, 0.6))
                )
        finally:
            capture.release()

        if not center_means:
            raise ValueError(f"No readable frames found in AVI video: {file_path}")

        video_series.append((file_path, np.asarray(center_means), fps))

    use_seconds = all(
        np.isfinite(fps) and fps > 0 for _, _, fps in video_series
    )

    channel_plots = (
        ("Blue", 2, "Blues"),
        ("Green", 1, "Greens"),
        ("Red", 0, "Reds"),
    )
    fig, axes = plt.subplots(
        len(channel_plots),
        1,
        figsize=(12, 4 * len(channel_plots)),
        squeeze=False,
    )
    for axis, (channel_name, channel_index, color_map) in zip(
        axes.flat, channel_plots
    ):
        shades = plt.get_cmap(color_map)(
            np.linspace(0.45, 0.95, len(video_series))
        )
        for (file_path, center_means, fps), color in zip(video_series, shades):
            x_values = (
                np.arange(len(center_means)) / fps
                if use_seconds
                else np.arange(len(center_means))
            )
            axis.plot(
                x_values,
                center_means[:, channel_index],
                color=color,
                label=os.path.relpath(file_path, data_directory),
            )

        axis.set_title(f"{channel_name} channel")
        axis.set_xlabel("Time (seconds)" if use_seconds else "Frame")
        axis.set_ylabel("Center ROI intensity (8-bit)")
        axis.set_ylim(0, 255)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Video")

    fig.suptitle("Center ROI RGB intensity over time by video")
    fig.tight_layout()
    if show:
        plt.show()

# P-value adjustment function
def _adjust_p_values(p_values):
    """Apply the Benjamini-Hochberg correction to a sequence of p-values."""
    p_values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p_values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if not len(valid):
        return adjusted

    order = valid[np.argsort(p_values[valid])]
    ranked = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    return adjusted


def compare_screen_recordings(df):
    """Compare matching screens across displays or individual AVI recordings."""
    metrics = (
        "Red_Center",
        "Green_Center",
        "Blue_Center",
        "Luminance_Center",
        "Uniformity_Corner_pct",
    )
    required_columns = {"PC_ID", "Recording", "Screen", *metrics}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot compare recordings; missing columns {sorted(missing_columns)}."
        )

    recording_means = (
        df.groupby(["Screen", "PC_ID", "Recording"], as_index=False)[list(metrics)]
        .mean()
    )
    display_comparison = recording_means["PC_ID"].nunique() > 1
    comparison_column = "PC_ID" if display_comparison else "Recording"
    source_column = "Display" if display_comparison else "Recording"
    comparisons = []
    for screen, screen_df in recording_means.groupby("Screen", sort=False):
        source_ids = sorted(screen_df[comparison_column].unique())
        for metric in metrics:
            for source_a, source_b in combinations(source_ids, 2):
                values_a = screen_df.loc[
                    screen_df[comparison_column] == source_a, metric
                ].to_numpy()
                values_b = screen_df.loc[
                    screen_df[comparison_column] == source_b, metric
                ].to_numpy()
                result = {
                    "Screen": screen,
                    "Metric": metric,
                    "Comparison": source_column,
                    "Source_A": source_a,
                    "Source_B": source_b,
                    "N_A": len(values_a),
                    "N_B": len(values_b),
                    "Mean_A": values_a.mean(),
                    "Mean_B": values_b.mean(),
                    "Difference_B_minus_A": values_b.mean() - values_a.mean(),
                    "P_Value": np.nan,
                    "Q_Value": np.nan,
                    "Significant": False,
                    "Result": "Insufficient independent recordings",
                }
                if len(values_a) >= 2 and len(values_b) >= 2:
                    test = ttest_ind(values_a, values_b, equal_var=False)
                    result["P_Value"] = float(test.pvalue)
                    result["Result"] = "Tested"
                comparisons.append(result)

    comparison_df = pd.DataFrame(comparisons)
    if comparison_df.empty:
        return comparison_df

    for _, metric_indices in comparison_df.groupby("Metric").groups.items():
        indices = list(metric_indices)
        p_values = comparison_df.loc[indices, "P_Value"].to_numpy()
        q_values = _adjust_p_values(p_values)
        comparison_df.loc[indices, "Q_Value"] = q_values
        for index, q_value in zip(indices, q_values):
            if not np.isfinite(comparison_df.loc[index, "P_Value"]):
                if comparison_df.loc[index, "Result"] == "Tested":
                    comparison_df.loc[index, "Result"] = "Not estimable"
                continue

            is_significant = q_value < 0.05
            comparison_df.loc[index, "Significant"] = is_significant
            comparison_df.loc[index, "Result"] = (
                "Significant" if is_significant else "Not significant"
            )

    return comparison_df

# Function to plot screen measurements across recordings and annotate significant results.
def plot_comparison(df, comparison_df=None, show=True):
    """Plot screen measurements across recordings and annotate significant results."""
    if df.empty:
        print("No AVI video data found to plot.")
        return

    screen_order = list(df["Screen"].drop_duplicates())
    if "Step" in df.columns:
        screen_order = (
            df.sort_values("Step")["Screen"].drop_duplicates().tolist()
        )
    group_column = "PC_ID" if df["PC_ID"].nunique() > 1 else "Recording"
    display_ids = sorted(df[group_column].unique())
    display_colors = plt.get_cmap("tab10")
    recording_means = df.groupby(
        ["Screen", "PC_ID", "Recording"], as_index=False
    ).mean(numeric_only=True)
    metrics = (
        ("Red_Center", "Center RGB intensity (8-bit)"),
        ("Green_Center", "Center RGB intensity (8-bit)"),
        ("Blue_Center", "Center RGB intensity (8-bit)"),
        ("Luminance_Center", "Center luminance (8-bit)"),
        ("Uniformity_Corner_pct", "Corner / center luminance (%)"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), squeeze=False)
    for axis, (metric, ylabel) in zip(axes.flat, metrics):
        for display_index, display_id in enumerate(display_ids):
            display_df = recording_means[
                recording_means[group_column] == display_id
            ]
            positions = []
            means = []
            errors = []
            for screen_index, screen in enumerate(screen_order):
                values = display_df.loc[display_df["Screen"] == screen, metric]
                if values.empty:
                    continue
                x_position = screen_index + (
                    display_index - (len(display_ids) - 1) / 2
                ) * 0.16
                positions.extend([x_position] * len(values))
                means.append((screen_index, x_position, values.mean()))
                errors.append(values.std(ddof=1) if len(values) > 1 else 0)
                axis.scatter(
                    [x_position] * len(values),
                    values,
                    color=display_colors(display_index % 10),
                    alpha=0.45,
                    s=24,
                )

            if means:
                axis.errorbar(
                    [row[1] for row in means],
                    [row[2] for row in means],
                    yerr=errors,
                    color=display_colors(display_index % 10),
                    marker="o",
                    linewidth=1.5,
                    capsize=3,
                    label=display_id,
                )

        axis.set_ylabel(ylabel)
        axis.set_title(metric.replace("_", " "))
        axis.set_xticks(range(len(screen_order)), screen_order, rotation=60, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    axes[2, 1].axis("off")
    if comparison_df is not None and not comparison_df.empty:
        luminance_tests = comparison_df[
            (comparison_df["Metric"] == "Luminance_Center")
            & comparison_df["Significant"]
        ]
        if not luminance_tests.empty and df["PC_ID"].nunique() > 1:
            significant_screens = set(luminance_tests["Screen"])
            luminance_axis = axes[1, 1]
            y_min, y_max = luminance_axis.get_ylim()
            y_span = y_max - y_min or 1
            for screen_index, screen in enumerate(screen_order):
                if screen in significant_screens:
                    luminance_axis.text(
                        screen_index,
                        y_max + y_span * 0.025,
                        "*",
                        ha="center",
                        va="bottom",
                        color="red",
                        fontsize=14,
                    )
            luminance_axis.set_ylim(y_min, y_max + y_span * 0.12)
            luminance_axis.set_title(
                "Center luminance (* significant difference between displays)"
            )

    fig.tight_layout()
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_directory",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "screen_data"),
        help="Directory containing AVI videos or PC_* subfolders.",
    )
    args = parser.parse_args([] if "ipykernel" in sys.modules else None)

    start_time = time.time()
    df_results = analyze_all_computers(args.data_directory)
    print(df_results)

    comparison_df = compare_screen_recordings(df_results)
    if comparison_df.empty:
        print(
            "\nNo file pairs contain the same screen state, so no comparisons "
            "were possible."
        )
    else:
        print(
            "\nScreen comparisons (Welch's t-test on independent recording "
            "means; Benjamini-Hochberg correction, q < 0.05):"
        )
        print(
            comparison_df[
                [
                    "Screen",
                    "Metric",
                    "Comparison",
                    "Source_A",
                    "Source_B",
                    "N_A",
                    "N_B",
                    "Difference_B_minus_A",
                    "P_Value",
                    "Q_Value",
                    "Result",
                ]
            ].to_string(index=False)
        )
        if (comparison_df["Result"] == "Insufficient independent recordings").any():
            print(
                "Significance was not tested where either comparison group has "
                "fewer than two independent recordings for that screen. Frame "
                "samples are not counted as independent recordings. To test "
                "differences between displays, put repeated recordings for "
                "each display in separate PC_* folders."
            )

    plot_video_time_series(args.data_directory, show=False)
    plot_comparison(df_results, comparison_df, show=False)
    plt.show()
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
