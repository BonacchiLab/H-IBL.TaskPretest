""" This script analyzes AVI videos for screen calibration, focusing on RGB intensity and spatial uniformity.
It is directly compatible with the bonsai workflow for screen calibration, but can also be run as a standalone script.
It analyze RGB intensity and spatial uniformity in screen-calibration AVI videos.

Each video should show a single screen. Videos may be placed directly in the data
directory or in subfolders named PC_01, PC_02, and so on.
"""

import argparse
import glob
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    if frame_count == 0:
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


def analyze_all_computers(data_directory):
    """Analyze AVI videos in PC_* subfolders or directly in the given directory."""
    if not os.path.isdir(data_directory):
        raise FileNotFoundError(f"Data directory does not exist: {data_directory}")

    pc_folders = sorted(
        (
            folder
            for folder in glob.glob(os.path.join(data_directory, "PC_*"))
            if os.path.isdir(folder)
        ),
        key=str.lower,
    )
    if not pc_folders:
        pc_folders = [data_directory]

    results = []
    for folder in pc_folders:
        pc_id = (
            os.path.basename(folder)
            if os.path.normcase(folder) != os.path.normcase(data_directory)
            else "Single_Folder"
        )
        avi_files = sorted(
            (
                os.path.join(folder, name)
                for name in os.listdir(folder)
                if name.lower().endswith(".avi")
                and os.path.isfile(os.path.join(folder, name))
            ),
            key=str.lower,
        )

        for file_path in avi_files:
            results.append(
                {
                    "PC_ID": pc_id,
                    "Screen": os.path.splitext(os.path.basename(file_path))[0],
                    **analyze_video(file_path),
                }
            )

    return pd.DataFrame(results)


def plot_comparison(df):
    """Plot center luminance and corner-to-center uniformity by computer."""
    if df.empty:
        print("No AVI video data found to plot.")
        return

    neutral_df = df[df["Screen"].str.contains("neutral", case=False, na=False)]
    if neutral_df.empty:
        neutral_df = df

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(
        neutral_df["PC_ID"],
        neutral_df["Luminance_Center"],
        color="gray",
        edgecolor="black",
    )
    plt.axhline(
        neutral_df["Luminance_Center"].mean(),
        color="red",
        linestyle="--",
        label="Global Mean",
    )
    plt.title("Neutral Background Luminance (Center)")
    plt.ylabel("Decoded RGB intensity (8-bit)")
    plt.xticks(rotation=45)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(
        neutral_df["PC_ID"],
        neutral_df["Uniformity_Corner_pct"],
        color="skyblue",
        edgecolor="black",
    )
    plt.axhline(100, color="green", linestyle=":", label="100% Homogeneous")
    plt.title("Screen Uniformity (Corner / Center %)")
    plt.ylabel("Luminance Ratio (%)")
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
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
    plot_comparison(df_results)
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
