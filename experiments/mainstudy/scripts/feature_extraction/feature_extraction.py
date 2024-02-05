#!/usr/bin/env python

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from empkins_io.datasets.d03.macro_ap01 import MacroStudyTsstDataset
from empkins_macro.feature_extraction import extract_expert_features, extract_generic_features
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from stresspose_analysis.feature_extraction.utils import (
    load_expert_feature_dict,
    load_generic_feature_dict,
    remove_na,
)

deploy_type = "hpc"
system = "xsens"

root_path = Path("../..")

config_dict = json.load(root_path.joinpath("config.json").open(encoding="utf-8"))

feature_dict_path = root_path.joinpath("params/feature_dicts_tsst")
output_path = root_path.joinpath("feature_export/movement_features")
output_path.mkdir(exist_ok=True)

base_path = Path(config_dict[deploy_type]["base_path"])
dataset = MacroStudyTsstDataset(base_path=base_path, use_cache=True)

sampling_rate = 60  # Hz

threshold_gyr = 5  # deg2/s2
window_sec_gyr = 0.5  # sec
overlap_percent_gyr = 0.5  # %

threshold_vel = 5e-5  # m2/s2
window_sec_vel = 0.5  # sec
overlap_percent_vel = 0.5  # %

distance_thres = 0.2  # m

generic_feature_dict = load_generic_feature_dict(feature_dict_path)
expert_feature_dict = load_expert_feature_dict(
    feature_dict_path,
    sampling_rate_hz=sampling_rate,
    threshold_gyr=threshold_gyr,
    window_sec_gyr=window_sec_gyr,
    overlap_percent_gyr=overlap_percent_gyr,
    threshold_vel=threshold_vel,
    window_sec_vel=window_sec_vel,
    overlap_percent_vel=overlap_percent_vel,
    distance_thres=distance_thres,
)


def process_subset(
    subset: MacroStudyTsstDataset, expert_feature_params, generic_feature_params
) -> Tuple[Tuple[str, str], pd.DataFrame]:
    subject_id = subset.index["subject"][0]
    condition = subset.index["condition"][0]
    mocap_data = subset.mocap_data

    generic_features = extract_generic_features(mocap_data, generic_feature_params, system="xsens")
    expert_features = extract_expert_features(mocap_data, expert_feature_params, system="xsens")
    return_data = pd.concat([generic_features, expert_features])

    return (subject_id, condition), return_data


num_processes = -1
parallel = Parallel(n_jobs=num_processes, return_as="generator")

group_levels = ["subject", "condition"]

dataset = dataset.groupby(group_levels)
results = tqdm(
    parallel(delayed(process_subset)(subset, expert_feature_dict, generic_feature_dict) for subset in dataset),
    total=len(dataset),
)

result_dict = dict(results)
print("Done extracting features!")

# Data Concatenation
movement_data_total = pd.concat(result_dict, names=group_levels)
movement_data_total = remove_na(movement_data_total)

movement_data_cleaned = movement_data_total.unstack(group_levels)
# Drop features that are NaN for any subject
movement_data_cleaned = movement_data_cleaned.dropna(how="any", axis=0)
# Drop features that are constant (e.g., 0) for all subjects
std_mask = movement_data_cleaned.std(axis=1) != 0
movement_data_cleaned = movement_data_cleaned.loc[std_mask]

# Bring dataframe back in original format
movement_data_cleaned = movement_data_cleaned.stack(group_levels)
movement_data_cleaned = movement_data_cleaned.reorder_levels(movement_data_total.index.names).sort_index()


# Export Data
movement_data_total.to_csv(output_path.joinpath("movement_features.csv"))
movement_data_cleaned.to_csv(output_path.joinpath("movement_features_cleaned.csv"))
movement_data_cleaned.to_csv(output_path.joinpath("movement_features_for_classification.csv"))
