# stresspose-analysis

![GitHub](https://img.shields.io/github/license/empkins/stresspose-analysis)
[![Lint](https://github.com/empkins/stresspose-analysis/actions/workflows/test-and-lint.yml/badge.svg)](https://github.com/empkins/stresspose-analysis/actions/workflows/test-and-lint.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/empkins/stresspose-analysis)

This repository contains the code for the project "Effect of acute psychosocial stress on body posture and movements".

This package contains various helper functions to work with the dataset 
(including [`tpcp`](https://github.com/mad-lab-fau/tpcp) `Dataset` representations) and to process the recorded data.
Additionally, it contains analysis experiments performed with the dataset to reproduce all results, figures, etc.


## Repository Structure

**The repository is structured as follows:**

```bash
├── stresspose_analysis/            # `stresspose-analysis` Python package with helper functions
└── experiments/                    # Folder with conducted analysis experiments; each experiment has its own subfolder
    ├── pilotstudy/                 # Data Analysis for the Pilot Study
    │   ├── feature_export/         # Exported motion features
    │   ├── notebooks/              # Notebooks for data processing, data analysis, classification experiments, plotting, etc. in subfolders
    │   ├── output/                 # Output of classification experiments (pickled models, input features, etc.)
    │   ├── params/                 # Hyperparameter settings for classification experiments
    │   ├── results/                # Analysis results
    │   └── config.json/            # Config file containing location of dataset (ignored because path depend on local configurations)
    └── mainstudy/                  # Data Analysis for the Main Study
        ├── feature_export/         # Exported motion features
        ├── notebooks/              # Notebooks for data processing, data analysis, classification experiments, plotting, etc. in subfolders
        ├── output/                 # Output of classification experiments (pickled models, input features, etc.)
        ├── params/                 # Hyperparameter settings for classification experiments
        ├── results/                # Analysis results
        ├── scripts/                # Scripts for feature extraction and classification experiments
        └── config.json/            # Config file containing location of dataset (ignored because path depend on local configurations)
```

## Installation
### Dataset
In order to run the code, first download the desired dataset(s) from OSF:
   * [Pilot Study](https://osf.io/qvzdg/)
   * [Main Study](https://osf.io/va6t3/)

Then, create a file named `config.json` in the respective `experiment` subfolders 
(`/experiments/pilotstudy` or `/experiments/mainstudy`) folder with the following content:
```json
{
  "name-of-deployment <e.g., local>": {
    "base_path": "<path-to-dataset>"
  }
}
```

This config file is parsed by all notebooks to extract the path to the dataset.   
**NOTE**: This file is ignored by git because the path to the dataset depends on the local configuration!

### Code
If you want to use this package to reproduce the analysis results the best way is to clone the repository and install
the package via [poetry](https://python-poetry.org).
For that, open a terminal and run the following commands:

```bash
git clone git@github.com:empkins/stresspose-analysis.git
cd stresspose-analysis
poetry install # alternative: pip install .
```

This creates a new python venv in the `stresspose-analysis/.venv` folder. Next, register a new IPython kernel for the 
venv:
```bash
cd stresspose-analysis
poetry run poe register_ipykernel
```

Finally, go to the `experiments` folder and run the Jupyter Notebooks to reproduce all data processing steps (see below).

## Experiments

### Machine learning-based detection of acute psychosocial stress from body posture and movements

This analysis was performed for the paper "Machine learning-based detection of acute psychosocial stress from body 
posture and movements", currently under review at _Scientific Reports_.

#### Notebook Processing Order
To run the data processing and analysis pipeline, we recommend executing the notebooks in the following order:
1. Pilot Study:
   1. Data Processing and Feature Extraction:
      1. Motion Features: Run `data_processing/Motion_Data_Preprocessing.ipynb` -> `data_processing/Motion_Data_Feature_Extraction.ipynb`
      2. Self-reports: Run `data_processing/Questionnaire_Processing.ipynb`
      3. Saliva Data: Run `data_processing/Saliva_Processing.ipynb`
   2. Classification Experiments: Run `classification/Classification_General.ipynb` and `classification/Classification_Talk.ipynb`
   3. Data Analysis:
      1. Saliva Data: Run `stress_response/Saliva_Analysis.ipynb`
      2. Self-reports: Run `stress_response/State_Questionnaire_Analysis.ipynb`
      3. Motion Features (statistical): Run `motion_features/Motion_Feature_Overview.ipynb` and `motion_features/Motion_Feature_Analysis.ipynb`
      4. Classification Results: Run `classification/Analysis_Classification_General.ipynb` and `classification/Analysis_Classification_Talk.ipynb`
   4. General Plots (plots used in papers, etc.): Run `plotting/Teaserfigure_Plots.ipynb`, `plotting/Sensor_Grid_Plots.ipynb`, and `plotting/Filter_Pipeline_Plots.ipynb`
2. Main Study:
   1. Data Processing and Feature Extraction:
      1. Motion Features: Run `movement_data/feature_extraction/Feature_Extraction.ipynb` and `movement_data/feature_extraction/Feature_Extraction_per_Phase.ipynb` (alternative: run the **scripts** `scripts/feature_extraction.py` and `scripts/feature_extraction_per_phase.py`) 
      2. Self-reports: Run `questionnaires/Questionnaire_Processing.ipynb` 
      3. Saliva Data: Run `biomarker/Saliva_Processing.ipynb`
   2. Classification Experiments: Run the **scripts** (not notebooks!) `scripts/classification.py` and `scripts/classification_rf.py`.
      Due to the long runtime of the classification experiments, only a subset of classification pipeline permutations are executed in one script. Thus, the type of feature selection and scaler can be passed as environment variables.
      See the scripts for more information.
   3. Data Analysis:
      1. Saliva Data: Run `analysis/stress_response/Saliva_Analysis.ipynb`
      2. Self-reports: Run `analysis/stress_response/State_Questionnaire_Analysis.ipynb`
      3. Motion Features (statistical): Run `analysis/movement_data/Movement_Feature_Overview.ipynb`, `analysis/movement_data/Movement_Feature_Analysis.ipynb`, and `analysis/movement_data/Movement_Feature_Analysis_per_Phase.ipynb`
      4. Classification Results: Run `analysis/classification/Analysis_Classification_General.ipynb` and `analysis/classification/Analysis_Classification_Per_Phase.ipynb`
   
