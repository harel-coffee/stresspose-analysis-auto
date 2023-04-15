"""Module containing utility functions for feature extraction."""
import json
from ast import literal_eval

import pandas as pd
from empkins_macro.utils._types import path_t


def load_generic_feature_dict(folder_path: path_t) -> dict:
    """Load a generic feature dictionary from a JSON file.

    Args:
        folder_path: Path to the JSON file.

    Returns
    -------
        A dictionary of features.
    """
    return json.load(folder_path.joinpath("generic_feature_dict.json").open(encoding="utf-8"))


def load_expert_feature_dict(folder_path: path_t, sampling_rate_hz: float) -> dict:
    """Load a JSON dictionary containing parameters for expert feature extraction.

    Parameters
    ----------
    folder_path : :class:`~pathlib.Path` or str
        Path to the JSON file.
    sampling_rate_hz : float
        Sampling rate of the data in Hz.

    Returns
    -------
    dict
        expert feature parameter dictionary
    """
    expert_feature_dict = json.load(folder_path.joinpath("expert_feature_dict.json").open(encoding="utf-8"))
    # convert dict to str in order to replace sampling_rate placeholder with actual sampling rate of data
    expert_feature_dict = str(expert_feature_dict)
    expert_feature_dict = expert_feature_dict.replace("'<sampling_rate>'", str(sampling_rate_hz))
    # convert string back to dict
    return literal_eval(expert_feature_dict)


def remove_na(data: pd.DataFrame) -> pd.DataFrame:
    """Remove all features with NaN values.

    This function removes all features where feature values are NaN for all subjects and conditions.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        motion capture features

    Returns
    -------
    :class:`~pandas.DataFrame`
        motion capture features with removed rows with NaN values

    """
    index_order = data.index.names
    stack_levels = ["subject", "condition"]
    if "phase" in data.index.names:
        stack_levels += ["phase"]

    data = data.unstack(stack_levels).dropna(how="all").stack(stack_levels)

    data = data.reorder_levels(index_order).sort_index()
    return data
