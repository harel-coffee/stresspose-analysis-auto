from typing import Optional, Sequence

import pandas as pd
from biopsykit.utils.dataframe_handling import add_space_to_camel

_phase_map = {
    "talk": "Interview",
    "math": "Mental Arithmetics",
}

_channel_map = {
    "acc": "Acc.",
    "ang_vel": "Ang. Vel.",
    "gyr": "Ang. Vel.",
    "vel": "Velocity",
    "rot": "Rotation",
}

_metric_map = {
    "mean_duration_sec": "Mean Duration (s)",
    "std_duration_sec": "Std. Duration (s)",
    "max_duration_sec": "Max. Duration (s)",
    "cov": "CoV",
    "abs_energy": "Abs. Energy",
    "entropy": "Entropy",
    "mean": "Mean",
    "ratio_percent": "Ratio (%)",
    "count_per_min": "Counts per Minute",
    "zero_crossing": "Zero Crossings",
    "max_val": "Max. Value",
    "std": "Std. Dev.",
    "fft_aggregated_centroid": "FFT Centroid",
    "fft_aggregated_skew": "FFT Skewness",
    "fft_aggregated_variance": "FFT Variance",
    "fft_aggregated_kurtosis": "FFT Kurtosis",
    "fft_aggregated_kurt": "FFT Kurtosis",
}

_type_map = {
    "mean": "Mean",
    "std": "Std. Dev.",
    "cov": "CoV",
    "max_val": "Max. Value",
    "abs_energy": "Abs. Energy",
    "entropy": "Entropy",
    "fft_aggregated_skew": "FFT Aggregated Skewness",
    "fft_aggregated_centroid": "FFT Aggregated Centroid",
    "fft_aggregated_kurtosis": "FFT Aggregated Kurtosis",
    "static_periods": "Static Periods",
    "zero_crossing": "Zero Crossings",
}

_axis_map = {"norm": "L2-norm", "x": "x-axis", "y": "y-axis", "z": "z-axis"}


def add_concat_feature_name_to_index(data: pd.DataFrame) -> pd.DataFrame:

    data_wide = data.unstack(["subject", "condition"])
    index_names = data_wide.index.names
    index = pd.Index(["-".join(i) for i in data_wide.index], name="feature_concat")
    data_wide = data_wide.assign(feature_concat=index)
    data_wide = data_wide.set_index("feature_concat", append=True)
    data_wide = data_wide.stack(["subject", "condition"]).reorder_levels(
        ["subject", "condition", *index_names] + ["feature_concat"]
    )
    return data_wide


def drop_multiindex(data: pd.DataFrame, levels_in: Optional[Sequence[str]] = None) -> pd.DataFrame:
    # levels to keep
    if levels_in is None:
        levels_in = ["subject", "condition", "feature_concat"]
    # levels to drop
    levels_out = [i for i in data.index.names if i not in levels_in]
    data = data.reset_index(level=levels_out, drop=True)
    return data


def add_multiindex_to_stats_results(data_stats: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    index = data.unstack(["subject", "condition"]).index
    multiindex = pd.DataFrame(list(index), columns=index.names)
    multiindex = multiindex.set_index("feature_concat")
    data_stats = data_stats.join(multiindex)

    index_levels = list(data_stats.drop(columns=["W-val", "p-corr", "hedges"]).columns)
    data_stats = data_stats.set_index(index_levels, append=True)
    return data_stats


def rename_motion_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.rename(
        lambda s: add_space_to_camel(s).replace("_", r" \&").replace("Spine3", "Chest"),
        level="body_part",
    )
    data = data.replace("%", r" \%")
    data = (
        data.rename(str.capitalize, level="feature_type")
        .rename(_channel_map, level="channel")
        .rename(_metric_map, level="metric")
        .rename(_type_map, level="type")
        .rename(_axis_map, level="axis")
    )
    if "phase" in data.index.names:
        data = data.rename(_phase_map, level="phase")
    return data
