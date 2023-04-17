# stresspose-analysis

![GitHub](https://img.shields.io/github/license/empkins/stresspose-analysis)
[![Lint](https://github.com/empkins/stresspose-analysis/actions/workflows/test-and-lint.yml/badge.svg)](https://github.com/empkins/stresspose-analysis/actions/workflows/test-and-lint.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/empkins/stresspose-analysis)

This repository contains the code for the paper "StressPose: Detection of Acute Psychosocial Stress from Body 
Posture and Movements using Machine Learning".

This package contains various helper functions to work with the dataset 
(including [`tpcp`](https://github.com/mad-lab-fau/tpcp) `Dataset` representations) and to process the recorded data.
Additionally, it contains analysis experiments performed with the dataset to reproduce all results, figures, etc.


## Repository Structure

**The repository is structured as follows:**

```bash
├── stresspose_analysis/                                    # `stresspose-analysis` Python package with helper functions
└── experiments/                                            # Folder with conducted analysis experiments; each experiment has its own subfolder
    └── 2023_02_stress_pose_imwut/                          # Analysis for the 2023 IMWUT Paper (see below)
        ├── feature_export/                                 # Exported motion features
        ├── notebooks/                                      # Notebooks for data processing, data analysis, classification experiments, plotting, etc. in subfolders
        ├── output/                                         # Output of classification experiments (pickled models, input features, etc.)
        ├── params/                                         # Hyperparameter settings for classification experiments
        ├── results/                                        # Analysis results
        └── config.json/                                    # Config file containing location of dataset (ignored because path depend on local configurations)
```

## Installation
### Dataset
In order to run the code, first download the StressPose Dataset, e.g. from [OSF](https://osf.io/qvzdg/). 
Then, create a file named `config.json` in the `/experiments/2023_02_stress_pose_imwut` folder with the following 
content:
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
If you want to use this package to reproduce the analysis results then clone the repository and install the 
package via [poetry](https://python-poetry.org). For that, open a terminal and run the following commands:

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

### 2023 IMWUT Paper (`2023_stress_pose_imwut`)

This analysis was performed for the paper "StressPose: Detection of Acute Psychosocial Stress from Body Posture and 
Movements using Machine Learning", currently under review.

#### Notebook Processing Order
To run the data processing and analysis pipeline, we recommend executing the notebooks in the following order:
1. Data Processing and Feature Extraction:
   1. Motion Features: Run `data_processing/Motion_Data_Preprocessing.ipynb` -> `data_processing/Motion_Data_Feature_Extraction.ipynb`
   2. Self-reports: Run `data_processing/Questionnaire_Processing.ipynb`
   3. Saliva Data: Run `data_processing/Saliva_Processing.ipynb`
2. Classification Experiments: Run `classification/Classification_General.ipynb`
3. Data Analysis:
   1. Self-reports: Run `stress_response/State_Questionnaire_Classification_Experiments.ipynb`
   2. Saliva Data: Run `stress_response/Saliva_Classification_Experiments.ipynb`
   3. Motion Features (statistical): Run `motion_features/Motion_Feature_Overview.ipynb` and `motion_features/Motion_Feature_Analysis.ipynb`
   4. Classification Results: Run `classification/Analysis_Classification_General.ipynb`
4. General Plots (plots used in papers, etc.): Run `plotting/Teaserfigure_Plots.ipynb`, `plotting/Sensor_Grid_Plots.ipynb`, and `plotting/Filter_Pipeline_Plots.ipynb`
