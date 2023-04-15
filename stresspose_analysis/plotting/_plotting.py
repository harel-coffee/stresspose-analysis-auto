"""Module for plotting macro data."""
from copy import deepcopy
from typing import Sequence, Tuple

import pandas as pd
from biopsykit.plotting import multi_feature_boxplot
from biopsykit.stats import StatsPipeline
from matplotlib import pyplot as plt

__all__ = ["plot_motion_features"]

# for some reason, we need to rename the features to shorter names before plotting,
# otherwise tight_layout will fail
# TODO: implement a generic function that renames long feature names
_rename_map = {
    "generic-TotalBody-vel-mean-mean-norm": "tb-vel-mean",
    "generic-Trunk-vel-mean-mean-norm": "trunk-vel-mean",
    "generic-Head-rot-zero_crossing-zero_crossing-z": "head-rot-zcross",
    "generic-RightHand-ang_vel-abs_energy-abs_energy-norm": "hand-angvel-energy",
    "expert-Trunk-vel-static_periods-max_duration_sec-norm": "trunk-sp-max",
    "expert-UpperExtremities-vel-static_periods-max_duration_sec-norm": "ue-sp-max",
    "expert-Head-vel-static_periods-mean_duration_sec-norm": "head-sp-mean",
    "expert-UpperExtremities-vel-static_periods-mean_duration_sec-norm": "ue-sp-mean",
    "expert-Head-ang_vel-static_periods-ratio_percent-norm": "head-sp-ratio",
    "expert-Trunk-ang_vel-static_periods-ratio_percent-norm": "trunk-sp-ratio",
}

# selection of features and assignment in subplots
_selected_features = {
    "generic": {
        "vel_mean": ["tb-vel-mean", "trunk-vel-mean"],
        "rot_zero_crossing": ["head-rot-zcross"],
        "ang_vel_energy": ["hand-angvel-energy"],
    },
    "expert": {
        "sp_mean": ["head-sp-mean", "ue-sp-mean"],
        "sp_max": ["trunk-sp-max", "ue-sp-max"],
        "sp_ratio": ["head-sp-ratio", "trunk-sp-ratio"],
    },
}

# xtick labels in the result plot
_xticks_mapping = {
    "generic": {
        "vel_mean": ["Total Body", "Trunk"],
        "rot_zero_crossing": ["Head"],
        "ang_vel_energy": ["Right Hand"],
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
        "rot_zero_crossing": "Count [#]",
        "ang_vel_energy": "Energy [$(°/s)^2$]",
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
        "vel_mean": "Velocity - Mean of L2-norm",
        "rot_zero_crossing": "Rotation around z-axis - Zero Crossings",
        "ang_vel_energy": "Ang. Vel. - Energy of L2-norm",
    },
    "expert": {
        "sp_mean": "Velocity - Static Periods of L2-norm",
        "sp_max": "Ang. Vel. - Static Periods of L2-norm",
        "sp_ratio": "Ang. Vel. - Static Periods of L2-norm",
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
    rect = kwargs.pop("rect", (0, 0, 0.90, 1))
    axs = kwargs.pop("axs", None)
    if axs is None:
        fig, axs = plt.subplots(ncols=3, figsize=(12, 3))
    else:
        fig = axs[0].get_figure()

    data_rename = data.rename(_rename_map, level="feature_concat")
    data_rename = data_rename.reindex(list(_rename_map.values()), level="feature_concat")

    stats_pipeline = deepcopy(stats_pipeline)
    stats_pipeline.results["pairwise_tests"] = stats_pipeline.results["pairwise_tests"].rename(
        _rename_map, level="feature_concat"
    )

    features_flat = _selected_features.get(feature_type)

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
        legend_loc="center right",
        legend_orientation="vertical",
        stats_kwargs={"box_pairs": box_pairs, "pvalues": pvalues, "verbose": 0},
        axs=axs,
        **kwargs,
    )

    for ax, feature_group in zip(axs, features_flat):
        ax.set_xticklabels(_xticks_mapping[feature_type][feature_group], fontsize="medium")
        ax.set_ylabel(_ylabel_mapping[feature_type][feature_group])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-10, 3), useMathText=True)

        ax.set_title(_title_mapping[feature_type][feature_group], y=1.05)
        ax.set_xlabel(None)

    fig.tight_layout(rect=rect, w_pad=1, pad=0)

    return fig, axs
