import os
import pandas as pd
import json
import time
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns


# Start timer
start_time = time.perf_counter()

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
path_input = input(
	"Enter the path to a CSV file or folder "
	f"[default: {data_dir}]: "
).strip().strip('"')

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




end_time = time.perf_counter()
print(f"Processing time: {end_time - start_time} seconds")