import json
from ast import literal_eval
from typing import Optional

import pandas as pd
from empkins_io.utils._types import path_t


def load_generic_feature_dict(folder_path: path_t, suffix: Optional[str] = None) -> dict:
    """Load a generic feature dictionary from a JSON file.

    Args:
        folder_path: Path to the JSON file.

    Returns
    -------
        A dictionary of features.
    """
    filename = "generic_feature_dict.json"
    if suffix is not None:
        filename = f"generic_feature_dict_{suffix}.json"

    return json.load(folder_path.joinpath(filename).open(encoding="utf-8"))


def load_expert_feature_dict(folder_path: path_t, sampling_rate_hz: float, **kwargs) -> dict:
    expert_feature_dict = json.load(folder_path.joinpath("expert_feature_dict.json").open(encoding="utf-8"))
    # convert dict to str in order to replace sampling_rate placeholder with actual sampling rate of data
    expert_feature_dict = str(expert_feature_dict)
    expert_feature_dict = expert_feature_dict.replace("'<sampling_rate>'", str(sampling_rate_hz))
    for key, val in kwargs.items():
        expert_feature_dict = expert_feature_dict.replace(f"'<{key}>'", str(val))
    # convert string back to dict
    return literal_eval(expert_feature_dict)


def remove_na(data: pd.DataFrame) -> pd.DataFrame:
    index_order = data.index.names
    stack_levels = ["subject", "condition"]
    if "phase" in data.index.names:
        stack_levels += ["phase"]

    data = data.unstack(stack_levels).dropna(how="all").stack(stack_levels)

    data = data.reorder_levels(index_order).sort_index()
    return data
