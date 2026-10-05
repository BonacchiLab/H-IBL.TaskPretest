'''This script is design to process RAW files (such as .CR2, .NEF, .ARW) 
captured by a camera, extracting the exact intensity values from each channel (RGB)
and comparing homogeneity across the image. This will allow for comparison between
different screens. '''

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rawpy
import time
import tifffile

start_time = time.time()  # Start the timer

def extract_raw_rgb_roi(file_path, roi_box=None):
    """Reads a RAW image (or synthetic TIFF) without applying Gamma or WB.

    roi_box: normalized tuple (x_min, y_min, x_max, y_max) ranging [0 to 1].
    """
    try:
        # 1. Try reading with rawpy for genuine camera RAW files (.CR2, .NEF, .ARW, native .DNG)
        with rawpy.imread(file_path) as raw:
            rgb_linear = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=False,
                user_wb=[1, 1, 1, 1],
                output_bps=16,
            )
    except Exception:
        # 2. Fallback for synthetic TIFF/DNG files generated for testing
        rgb_linear = tifffile.imread(file_path)

    h, w = rgb_linear.shape[0], rgb_linear.shape[1]

    if roi_box is None:
        # Default: Center region (40% to 60%)
        x_min, y_min = int(w * 0.4), int(h * 0.4)
        x_max, y_max = int(w * 0.6), int(h * 0.6)
    else:
        x_min, y_min = int(w * roi_box[0]), int(h * roi_box[1])
        x_max, y_max = int(w * roi_box[2]), int(h * roi_box[3])

    roi = rgb_linear[y_min:y_max, x_min:x_max]

    r_mean = np.mean(roi[:, :, 0])
    g_mean = np.mean(roi[:, :, 1])
    b_mean = np.mean(roi[:, :, 2])

    return r_mean, g_mean, b_mean


def analyze_all_computers(data_directory):
    """Processes images inside subfolders (PC_01, PC_02...) or directly inside data_directory."""
    results = []

    # Check if there are subfolders like PC_01, PC_02...
    pc_folders = sorted(glob.glob(os.path.join(data_directory, "PC_*")))

    # If no PC_* subfolders exist, process the main directory as a single folder
    if not pc_folders:
        pc_folders = [data_directory]

    valid_extensions = ("*.dng", "*.CR2", "*.NEF", "*.ARW", "*.tif", "*.tiff")

    for folder in pc_folders:
        pc_id = (
            os.path.basename(folder)
            if folder != data_directory
            else "Single_Folder"
        )

        raw_files = []
        for ext in valid_extensions:
            raw_files.extend(glob.glob(os.path.join(folder, ext)))

        raw_files = sorted(raw_files)

        for file_path in raw_files:
            screen_type = os.path.basename(file_path).split(".")[0]

            # 1. Center Measurement
            r_c, g_c, b_c = extract_raw_rgb_roi(
                file_path, roi_box=(0.4, 0.4, 0.6, 0.6)
            )

            # 2. Top-Left Corner Measurement (Spatial Uniformity)
            r_tl, g_tl, b_tl = extract_raw_rgb_roi(
                file_path, roi_box=(0.05, 0.05, 0.15, 0.15)
            )

            # Calculation of light falloff / Vignetting (Center vs Corner)
            lum_center = 0.2126 * r_c + 0.7152 * g_c + 0.0722 * b_c
            lum_corner = 0.2126 * r_tl + 0.7152 * g_tl + 0.0722 * b_tl
            uniformity_ratio = (
                (lum_corner / lum_center) * 100 if lum_center > 0 else 0
            )

            results.append({
                "PC_ID": pc_id,
                "Screen": screen_type,
                "Red_Center": r_c,
                "Green_Center": g_c,
                "Blue_Center": b_c,
                "Luminance_Center": lum_center,
                "Uniformity_Corner_pct": uniformity_ratio,
            })

    df = pd.DataFrame(results)
    return df


def plot_comparison(df):
    """Generates comparison plots across all computers."""
    if df.empty:
        print("No image data found to plot.")
        return

    neutral_df = df[df["Screen"].str.contains("neutral", case=False, na=False)]
    if neutral_df.empty:
        neutral_df = df  # Plot all if no specific 'neutral' label is found

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
    plt.ylabel("RAW Intensity (16-bit)")
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
    
elapsed_time = time.time() - start_time  # Calcula a diferença exata
print(f"Total processing time: {elapsed_time:.2f} seconds")


df_results = analyze_all_computers("dataset_test")
print(df_results)
plot_comparison(df_results)