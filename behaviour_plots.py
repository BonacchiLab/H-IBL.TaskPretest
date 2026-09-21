import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Start processing timer
start_time = time.perf_counter()

# Data input, handles both single CSV files and directories containing multiple CSV files.
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
path_input = input(
    "Enter the path to a CSV file or folder "
    f"[default: {data_dir}]: "
).strip().strip('"')

# Determine the target path based on user input or default data directory.
target_path = os.path.abspath(os.path.expanduser(path_input or data_dir))
if os.path.isfile(target_path):
    if not target_path.lower().endswith(".csv"):
        raise ValueError(f"The selected file is not a CSV: {target_path}")
    csv_paths = [target_path]
elif os.path.isdir(target_path):
    csv_paths = [
        os.path.join(folder, filename)
        for folder, _, filenames in os.walk(target_path)
        for filename in filenames
        if filename.lower().endswith(".csv")
    ]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files were found in {target_path}")
else:
    raise FileNotFoundError(f"The file or folder was not found: {target_path}")

# Provide feedback on the number of CSV files found. For debugging purposes.
if len(csv_paths) == 1:
    print(f"Found 1 CSV file: {csv_paths}")
else:
    print(f"Found {len(csv_paths)} CSV files.")



# Opens and concatenates all CSV files into a single DataFrame, handling potential read errors.
df_list = []

# Read each CSV file and append to the list
for f in csv_paths:
    try:
        temp_df = pd.read_csv(f, sep=None, engine='python')
        df_list.append(temp_df)
    except Exception as e:
        print(f"Error reading file {f}: {e}")

# Debugging: print the number of successfully read files
print(f"Successfully read {len(df_list)} out of {len(csv_paths)} CSV files.")

if not df_list:
    raise ValueError("No valid data could be extracted from the CSV files found.")

# Concatenate all DataFrames into a single DataFrame and clean column names.
df = pd.concat(df_list, ignore_index=True)
df.columns = df.columns.str.strip()

# Check for the presence of required columns (two for the pretest) and compute the 'sign' and 'contrast' values.
if 'position' in df.columns and 'grating' in df.columns:
    # Determine the sign from the X coordinate in the position value.
    # A negative value such as '-30' represents a left stimulus (-1); otherwise it is right (1).
    df['sign'] = df['position'].apply(lambda x: -1 if '-' in str(x) else 1)
    
    # Calculate delta contrast as a percentage (-100% to 100%).
    df['contrast'] = df['sign'] * df['grating'] * 100
else:
    raise KeyError("The 'position' or 'grating' columns were not found in the file.")

# Response mapping (keySide.keys) 
response_column = 'keySide.keys'

if response_column in df.columns:
    # Map: 'l' (right) -> 1, 's' (left) -> 0. Other responses become NaN and are discarded.
    df['choice_binary'] = df[response_column].apply(
        lambda x: 1 if str(x).strip().lower() == 'l' else (0 if str(x).strip().lower() == 's' else np.nan)
    )
    # Remove trials without a valid response.
    df = df.dropna(subset=['choice_binary'])
else:
    raise KeyError(f"The response column '{response_column}' was not found in your CSV file.")


# Mathematical definition of the psychometric function (logistic curve) 
def psychometric_function(x, bias, slope):
    return 100 / (1 + np.exp(-slope * (x - bias)))


# Plot generation
fig, ax = plt.subplots(figsize=(6, 5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

main_color = "#7E1010"
x_smooth = np.linspace(-32, 32, 200)

# Calculate the mean rightward-choice rate (%) and standard error of the mean by contrast.
stats = df.groupby('contrast')['choice_binary'].agg(['mean', 'sem']).reset_index()
stats['mean'] *= 100
stats['sem'] *= 100

x_data = stats['contrast'].values
y_data = stats['mean'].values
y_err = stats['sem'].values

# Fit a global logistic curve to the experimental data.
try:
    # Initial guess for [bias, slope].
    popt, _ = curve_fit(psychometric_function, x_data, y_data, p0=[0, 0.1], maxfev=5000)
    y_smooth = psychometric_function(x_smooth, *popt)
    ax.plot(x_smooth, y_smooth, color=main_color, lw=2.5, label='Psychometric fit')
    print(f"\nFit completed successfully:")
    print(f" -> Bias (target point of indifference): {popt[0]:.2f}%")
    print(f" -> Slope (subject sensitivity): {popt[1]:.4f}")
except Exception as e:
    print(f"\nCould not fit the logistic curve: {e}")

# Experimental plot points with the actual error bars.
ax.errorbar(x_data, y_data, yerr=y_err, fmt='o', color=main_color, 
            mec='w', mew=0.5, ms=7, label='Subject data')

# Configure axes and plot formatting.
ax.set_xlabel(r'$\Delta$ Contrast (%)', fontsize=11)
ax.set_ylabel('Rightward choices (%)', fontsize=11)
ax.set_ylim(-5, 105)
ax.set_xlim(-35, 35)

# Show every available contrast level on the x-axis.
x_ticks_clean = [-32, -16, -8, -4, -2, 0, 2, 4, 8, 16, 32]
ax.set_xticks(x_ticks_clean)
ax.set_xticklabels([str(t) for t in x_ticks_clean], rotation=45)

ax.legend(frameon=False, loc='upper left')
plt.tight_layout()
plt.show()

# Print elapsed data-processing time.
end_time = time.perf_counter()
print(f"Processing time: {end_time - start_time:.4f} seconds")