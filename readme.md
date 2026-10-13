[![PsychoPy](https://img.shields.io/badge/PsychoPy-v2026.1.2-blue)](https://www.psychopy.org/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)

## H-IBL Pretest

This repository contains the first pretest for the implementation of the Human International Brain Laboratory (H-IBL) task developed in PsychoPy.

The project aims to find the gratings that best recreate the behaviour as seen in the mice during the IBL perceptual decision-making as well as preserving the core experimental design.

This is an early development version intended for ongoing implementation, testing, and refinement. The codebase is expected to evolve as the task is validated and expanded. 

## Installation

Clone the repository:

```bash
git clone https://github.com/BonacchiLab/H-IBL.TaskPretest.git
```

Navigate to the project directory:

```bash
cd H-IBL.TaskPretest
```

Open the `.psyexp` file in PsychoPy Builder (recommended) or run the generated Python script to start the experiment.

> **Note:** Additional setup instructions and dependencies will be documented as the implementation progresses.


## Repository Structure & Overview

This repository contains the full code, experimental workflows, hardware calibration scripts, and data processing pipelines for the human adaptation of the **International Brain Laboratory (IBL)** perceptual decision-making paradigm.

```text
H-IBL.Task/
├── EyeTracking/                # Pupilometry and eye-tracking pipeline 
├── screen_calibration/         # Screen calibration scripts and photometric analysis
├── sound_files/                # Task audio stimuli (e.g., feedback noise burst)
├── Conditions_Tut.xlsx         # Stimulus conditions & Gabor parameters for the tutorial/practice phase
├── Conditions_parameters.xlsx  # Full task parameters (randomized Gabor phase, contrast levels)
├── Task_PreTest.psyexp         # PsychoPy Graphical Builder file for the pre-test task
├── Task_PreTest.py             # Main Python script running the task (includes looped tutorial)
├── Task_PreTest_lastrun.py     # Compiled PsychoPy execution file
├── behaviour_plots.py          # Behavioral data visualization (psychometric curves, choice bias)
├── requirements.txt            # Python dependencies (managed via uv)
└── requirements.in             # Core requirements file
