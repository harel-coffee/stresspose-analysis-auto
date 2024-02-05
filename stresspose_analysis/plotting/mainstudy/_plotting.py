"""Module for plotting macro data."""
from copy import deepcopy
from typing import Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import shap
from biopsykit.plotting import multi_feature_boxplot
from biopsykit.stats import StatsPipeline
from matplotlib import pyplot as plt

__all__ = ["plot_motion_features", "plot_motion_features_per_phase", "shap_feature_importances_plot"]

# for some reason, we need to rename the features to shorter names before plotting,
# otherwise tight_layout will fail
# TODO: implement a generic function that renames long feature names
_rename_map = {
    "generic-TotalBody-vel-mean-mean-norm": "tb-vel-mean",
    "generic-Trunk-vel-mean-mean-norm": "trunk-vel-mean",
    "generic-Head-acc-mean-mean-norm": "head-acc-mean",
    "generic-Head-gyr-entropy-entropy-norm": "head-gyr-entropy",
    "expert-Head-gyr-static_periods-mean_duration_sec-norm": "head-sp-mean",
    "expert-UpperExtremities-gyr-static_periods-mean_duration_sec-norm": "ue-sp-mean",
    "expert-Trunk-vel-static_periods-max_duration_sec-norm": "trunk-sp-max",
    "expert-UpperExtremities-vel-static_periods-max_duration_sec-norm": "ue-sp-max",
    "expert-Head-gyr-static_periods-ratio_percent-norm": "head-sp-ratio",
    "expert-Trunk-gyr-static_periods-ratio_percent-norm": "trunk-sp-ratio",
}

_rename_map_per_phase = {
    "math-generic-Head-gyr-mean-mean-norm": "math-head-gyr-mean",
    "math-generic-Head-gyr-cov-cov-norm": "math-head-cov-mean",
    "math-generic-Head-gyr-entropy-entropy-norm": "math-head-gyr-entropy",
    "talk-generic-Head-gyr-mean-mean-norm": "talk-head-gyr-mean",
    "talk-generic-Head-gyr-cov-cov-norm": "talk-head-cov-mean",
    "talk-generic-Head-gyr-entropy-entropy-norm": "talk-head-gyr-entropy",
    "math-expert-UpperExtremities-gyr-static_periods-mean_duration_sec-norm": "math-ue-sp-mean",
    "talk-expert-UpperExtremities-gyr-static_periods-mean_duration_sec-norm": "talk-ue-sp-mean",
    "math-expert-LeftHand_RightHand-gyr-static_periods-mean_duration_sec-norm": "math-hands-sp-mean",
    "talk-expert-LeftHand_RightHand-gyr-static_periods-mean_duration_sec-norm": "talk-hands-sp-mean",
}

# selection of features and assignment in subplots
_selected_features = {
    "generic": {
        "vel_mean": ["tb-vel-mean", "trunk-vel-mean"],
        "acc_mean": ["head-acc-mean"],
        "gyr_entropy": ["head-gyr-entropy"],
    },
    "expert": {
        "sp_mean": ["head-sp-mean", "ue-sp-mean"],
        "sp_max": ["trunk-sp-max", "ue-sp-max"],
        "sp_ratio": ["head-sp-ratio", "trunk-sp-ratio"],
    },
}

_selected_features_per_phase = {
    "generic": {
        "gyr_mean": ["math-head-gyr-mean", "talk-head-gyr-mean"],
        "gyr_cov": ["math-head-cov-mean", "talk-head-cov-mean"],
        "gyr_entropy": ["math-head-gyr-entropy", "talk-head-gyr-entropy"],
    },
    "expert": {
        "sp_mean": ["math-ue-sp-mean", "talk-ue-sp-mean", "math-hands-sp-mean", "talk-hands-sp-mean"],
        # "sp_max": ["trunk-sp-max", "ue-sp-max"],
    },
}

# xtick labels in the result plot
_xticks_mapping = {
    "generic": {
        "vel_mean": ["Total\nBody", "Trunk"],
        "acc_mean": ["Head"],
        "gyr_entropy": ["Head"],
    },
    "expert": {
        "sp_mean": ["Head", "Upper\nExtremities"],
        "sp_max": ["Trunk", "Upper\nExtremities"],
        "sp_ratio": ["Head", "Trunk"],
    },
}

# xtick labels in the result plot
_xticks_mapping_phase = {
    "generic": {
        "vel_mean": ["Total\nBody", "Trunk"],
        "acc_mean": ["Head"],
        "gyr_entropy": ["Head"],
    },
    "expert": {
        "sp_mean": ["Head", "Upper\nExtremities"],
        "sp_max": ["Trunk", "Upper\nExtremities"],
        "sp_ratio": ["Head", "Trunk"],
    },
}

# ylabels in the result plot
_ylabel_mapping = {
    "generic": {
        "vel_mean": "Velocity [m/s]",
        "acc_mean": "Acceleration [$m/s^2$]",
        "gyr_entropy": "Entropy [A.U.]",
    },
    "expert": {
        "sp_mean": "Mean Duration [s]",
        "sp_max": "Maximum Duration [s]",
        "sp_ratio": "Ratio [%]",
    },
}

# axes titles in the result plot
_title_mapping = {
    "generic": {
        "vel_mean": "Velocity\nMean",
        "acc_mean": "Acceleration\nMean",
        "gyr_entropy": "Ang. Vel.\nEntropy",
    },
    "expert": {
        "sp_mean": "Ang. Vel.\nStatic Periods",
        "sp_max": "Velocity\nStatic Periods",
        "sp_ratio": "Ang. Vel.\nStatic Periods",
    },
}


def plot_motion_features(
    data: pd.DataFrame, stats_pipeline: StatsPipeline, feature_type: str, **kwargs
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot motion features.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        feature data
    stats_pipeline : :class:`~biopsykit.stats.StatsPipeline`
        stats pipeline instance
    feature_type : str
        type of features to plot (generic or expert)
    **kwargs
        optional keyword arguments

    Returns
    -------
    :class:`~matplotlib.figure.Figure`
        figure
    list of :class:`~matplotlib.axes.Axes`
        axes

    """
    return _plot_motion_features(
        data, stats_pipeline, feature_type, rename_map=_rename_map, selected_featues=_selected_features, **kwargs
    )


def plot_motion_features_per_phase(
    data: pd.DataFrame, stats_pipeline: StatsPipeline, feature_type: str, **kwargs
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot motion features.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        feature data
    stats_pipeline : :class:`~biopsykit.stats.StatsPipeline`
        stats pipeline instance
    feature_type : str
        type of features to plot (generic or expert)
    **kwargs
        optional keyword arguments

    Returns
    -------
    :class:`~matplotlib.figure.Figure`
        figure
    list of :class:`~matplotlib.axes.Axes`
        axes

    """
    return _plot_motion_features(
        data,
        stats_pipeline,
        feature_type,
        _rename_map_per_phase,
        _selected_features_per_phase,
        **kwargs,
    )


def _plot_motion_features(
    data: pd.DataFrame,
    stats_pipeline: StatsPipeline,
    feature_type: str,
    rename_map: Dict[str, str],
    selected_features: Dict[str, Dict[str, str]],
    **kwargs,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot motion features.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        feature data
    stats_pipeline : :class:`~biopsykit.stats.StatsPipeline`
        stats pipeline instance
    feature_type : str
        type of features to plot (generic or expert)
    **kwargs
        optional keyword arguments

    Returns
    -------
    :class:`~matplotlib.figure.Figure`
        figure
    list of :class:`~matplotlib.axes.Axes`
        axes

    """
    rect = kwargs.pop("rect", (0, 0, 1.0, 0.90))
    axs = kwargs.pop("axs", None)
    if axs is None:
        fig, axs = plt.subplots(ncols=3, figsize=(7, 4))
    else:
        fig = axs[0].get_figure()

    data_rename = data.rename(rename_map, level="feature_concat")
    data_rename = data_rename.reindex(list(rename_map.values()), level="feature_concat")

    stats_pipeline = deepcopy(stats_pipeline)
    stats_pipeline.results["pairwise_tests"] = stats_pipeline.results["pairwise_tests"].rename(
        rename_map, level="feature_concat"
    )

    features_flat = selected_features.get(feature_type)

    box_pairs, pvalues = stats_pipeline.sig_brackets(
        "test", stats_effect_type="within", plot_type="multi", x="condition", subplots=True, features=features_flat
    )

    multi_feature_boxplot(
        data=data_rename,
        x="feature_concat",
        y="data",
        hue="condition",
        group="feature_concat",
        hue_order=["f-TSST", "TSST"],
        features=features_flat,
        legend_loc="upper center",
        legend_orientation="horizontal",
        stats_kwargs={"box_pairs": box_pairs, "pvalues": pvalues, "verbose": 0},
        tight_layout=False,
        axs=axs,
        **kwargs,
    )

    for ax, feature_group in zip(axs, features_flat):
        ax.set_xticklabels(_xticks_mapping[feature_type][feature_group], fontsize="medium")
        ax.set_ylabel(_ylabel_mapping[feature_type][feature_group])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-10, 3), useMathText=True)

        ax.set_title(_title_mapping[feature_type][feature_group], y=1.05)
        ax.set_xlabel(None)

    fig.tight_layout(rect=rect, w_pad=1, pad=0.1)

    return fig, axs


def shap_feature_importances_plot(shap_values: np.ndarray, features: pd.DataFrame, **kwargs) -> None:
    feature_names = [col.split("-")[2:-1] for col in features.columns]
    feature_names = [tuple(col[0:2] + col[3:]) for col in feature_names]
    feature_names = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(feature_names, names=["body_part", "channel", "metric"])
    )
    feature_names = feature_names.rename(
        {"gyr": "Ang. Vel.", "acc": "Acceleration", "vel": "Velocity", "pos": "Position"}, level="channel"
    )
    feature_names = feature_names.rename(
        {
            "ratio_percent": "S.P. Ratio",
            "mean_duration_sec": "S.P. Mean Dur.",
            "max_duration_sec": "S.P. Max. Dur.",
            "count_per_min": "S.P. #/min",
            "mean": "Mean",
            "std": "Std. Dev.",
            "cov": "CoV",
            "entropy": "Entropy",
            "abs_max": "Max. Value",
            "max_val": "Max. Value",
            "fft_aggregated_kurtosis": "FFT Kurt.",
        },
        level="metric",
    )
    feature_names = feature_names.rename({"LeftHand_RightHand": "Hands"}, level="body_part")

    feature_names = feature_names.index.tolist()
    feature_names = ["  –  ".join(f) for f in feature_names]

    shap.summary_plot(
        shap_values, features=features, plot_size=kwargs.get("plot_size", (12, 5)), feature_names=feature_names
    )
