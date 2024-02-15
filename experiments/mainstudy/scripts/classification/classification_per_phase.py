#!/usr/bin/env python3
import os
import warnings
from pathlib import Path

import biopsykit as bp
from biopsykit.classification.model_selection import SklearnPipelinePermuter
from biopsykit.classification.utils import prepare_df_sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold

from stresspose_analysis.classification.hyperparameter_search.mainstudy import (
    get_hyper_para_dict,
    get_hyper_search_dict,
    get_model_dict,
)
from stresspose_analysis.classification.utils import flatten_wide_format_column_names

env_vars = {key: val for key, val in os.environ.items() if key.startswith("PARAM__")}

random_state = int(env_vars.pop("PARAM__RANDOM_STATE", 0))

scaler = env_vars.get("PARAM__SCALER", None)
fsel = env_vars.get("PARAM__FSEL", None)

file_name = Path(__file__).with_suffix("").name

input_path = Path("../../")
output_path = Path("../../output/classification")
output_path.mkdir(exist_ok=True, parents=True)

feature_file = input_path.joinpath(
    "feature_export/movement_features/movement_features_per_phase_for_classification.csv"
)

print(f"Using feature file: {feature_file}")

data = bp.io.load_long_format_csv(feature_file)

levels_unstack = list(data.index.names)
for level in ["subject", "condition"]:
    levels_unstack.remove(level)
data_wide = data["data"].unstack(levels_unstack)
data_wide = flatten_wide_format_column_names(data_wide)


X, y, groups, group_keys = prepare_df_sklearn(data_wide, label_col="condition", print_summary=True)
num_subjects = len(group_keys)

model_dict = get_model_dict(scaler=scaler, fsel=fsel)
params_dict = get_hyper_para_dict(num_subjects=num_subjects)
hyper_search_dict = get_hyper_search_dict()

outer_cv = GroupKFold(5)
inner_cv = GroupKFold(5)

base_file_name = str(file_name)
for key, value in env_vars.items():
    if value is None:
        continue
    key_name = key.split("__")[-1].lower()
    base_file_name += f"_{key_name}_{value}"

base_file_name += "_pipeline_permuter.pkl"

# check if pipeline permuter already exists from previous run
permuter_file_path = output_path.joinpath(base_file_name)
if permuter_file_path.exists():
    print(f"Loading pre-fitted pipeline permuter from {permuter_file_path}.")
    pipeline_permuter = SklearnPipelinePermuter.from_pickle(permuter_file_path)
else:
    pipeline_permuter = SklearnPipelinePermuter(model_dict, params_dict, hyper_search_dict, random_state=random_state)

print(f"RANDOM STATE: {pipeline_permuter.random_state.get_state()[1][0]}")

data.to_csv(output_path.joinpath(f"{feature_file.stem}.csv"))

# fit all pipelines
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    pipeline_permuter.fit_and_save_intermediate(
        X,
        y,
        file_path=permuter_file_path,
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        groups=groups,
        use_cache=False,
    )

metric_summary = pipeline_permuter.metric_summary()
metric_summary = metric_summary.sort_values(by="mean_test_accuracy", ascending=False)

print("METRIC SUMMARY")
print(metric_summary[["mean_test_accuracy", "std_test_accuracy"]].head())

pipeline_permuter.to_pickle(permuter_file_path)
