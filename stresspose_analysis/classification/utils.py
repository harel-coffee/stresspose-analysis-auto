"""Module containing utility functions for the classification tasks."""
from typing import Optional, Sequence, Tuple

import pandas as pd
from biopsykit.classification.model_selection import SklearnPipelinePermuter
from biopsykit.utils.dataframe_handling import add_space_to_camel, snake_to_camel
from empkins_io.utils._types import str_t

from stresspose_analysis.data_wrangling import rename_motion_features


def flatten_wide_format_column_names(data: pd.DataFrame, col_index_name: Optional[str] = "feature") -> pd.DataFrame:
    """Flatten column names of wide-format data.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        wide-format data
    col_index_name : str, optional
        name of the new column index, default is ``"feature"``

    Returns
    -------
    :class:`~pandas.DataFrame`
        wide-format data with flattened column names

    """
    data.columns = ["-".join(col) for col in data.columns]
    data.columns.name = col_index_name
    return data


def feature_data_long_to_wide(data: pd.DataFrame, index_levels_out: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Convert long feature data to wide format.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        long feature data
    index_levels_out : list of str, optional
        new index levels for the wide-format data. Must be a subset of the current index levels. ``None`` to use
        the default levels ["subject", "condition"]

    Returns
    -------
    :class:`~pandas.DataFrame`
        wide feature data

    """
    if index_levels_out is None:
        index_levels_out = ["subject", "condition"]
    levels_unstack = list(data.index.names)
    for level in index_levels_out:
        levels_unstack.remove(level)
    data_wide = data["data"].unstack(levels_unstack)
    data_wide = flatten_wide_format_column_names(data_wide)

    return data_wide


def get_feature_counts(
    pipeline_permuter: SklearnPipelinePermuter,
    data: pd.DataFrame,
    pipeline: Tuple[str],
    index_levels: Optional[str_t] = None,
    feature_selection_key: Optional[str] = "reduce_dim",
    num_features: Optional[int] = None,
) -> pd.DataFrame:
    """Get the number of features selected by the feature selection algorithm of a specific pipeline.

    Parameters
    ----------
    pipeline_permuter : :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter`
        :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter` instance
    data : :class:`~pandas.DataFrame`
        dataframe with features
    pipeline : tuple of str
        selected pipeline from the PipelinePermuter
    index_levels : str or list of str, optional
        index levels to be used for the feature counts. If ``None``, this defaults to the levels
        ``["subject", "condition"]``
    feature_selection_key : str, optional
        key of the feature selection step in the pipeline. Default: ``"reduce_dim"``
    num_features : int, optional
        minimum number of features to be selected by the feature selection algorithm. If ``None``, all features
        are returned

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with feature counts

    """
    if index_levels is None:
        index_levels = ["subject", "condition"]
    if isinstance(index_levels, str):
        index_levels = [index_levels]
    best_estimator_summary = pipeline_permuter.best_estimator_summary()
    levels = list(data.index.names)
    for level in index_levels:
        levels.remove(level)

    best_pipeline = best_estimator_summary.loc[pipeline].iloc[0]

    list_features = []
    for pipeline in best_pipeline.pipeline:
        data_unstack = data.unstack(levels)["data"]
        data_transform = data_unstack.copy()
        for step, estimator in pipeline.steps:
            data_transform = estimator.transform(data_transform)
            data_transform = pd.DataFrame(data_transform, index=data_unstack.index)
            if hasattr(estimator, "get_support"):
                columns = data_unstack.columns[estimator.get_support()]
            else:
                columns = data_unstack.columns
            data_transform.columns = columns
            data_unstack = data_transform
            if step == feature_selection_key:
                features = list(data_unstack.columns)
                list_features.append(features)
                break

    features = [f for sublist in list_features for f in sublist]
    feature_counts = pd.DataFrame(features, columns=levels).value_counts()
    feature_counts = pd.DataFrame(feature_counts, columns=["Count"])
    if num_features is not None:
        feature_counts = feature_counts[feature_counts["Count"] >= num_features]
    return feature_counts


def get_number_features_per_fold(pipeline_permuter: SklearnPipelinePermuter, pipeline: Tuple[str]) -> pd.DataFrame:
    """Get the number of features per fold selected by the feature selection algorithm of a specific pipeline.

    Parameters
    ----------
    pipeline_permuter : :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter`
        :class:`~biopsykit.classification.model_selection.SklearnPipelinePermuter` instance
    pipeline : tuple of str
        selected pipeline from the PipelinePermuter

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with number of features per fold

    """
    best_estimator_summary = pipeline_permuter.best_estimator_summary()

    best_pipeline = best_estimator_summary.loc[pipeline].iloc[0]
    fsels = [pipe["reduce_dim"] for pipe in best_pipeline.pipeline]

    num_features = [fsel.n_features_ for fsel in fsels]
    num_features = pd.DataFrame(num_features, columns=["num_features"])
    num_features.index.name = "fold"
    return num_features


def feature_counts_to_latex(feature_counts: pd.DataFrame, **kwargs) -> str:
    """Convert feature counts to LaTeX table.

    Parameters
    ----------
    feature_counts : :class:`~pandas.DataFrame`
        feature counts, as returned by :func:`~stresspose_analysis.classification.utils.get_feature_counts`
    **kwargs
        additional keyword arguments passed to :meth:`~pandas.io.formats.style.Styler.to_latex`

    """
    kwargs.setdefault("sparse_index", False)
    kwargs.setdefault("hrules", True)
    kwargs.setdefault("position", "ht!")
    kwargs.setdefault("position_float", "centering")
    feature_counts = rename_motion_features(feature_counts)
    feature_counts.index = feature_counts.index.rename(
        [add_space_to_camel(snake_to_camel(idx)) for idx in feature_counts.index.names]
    )
    feature_counts = feature_counts.droplevel(["Feature Type", "Type"])
    feature_counts = feature_counts.rename({"Max. Duration (s)": "Stat. Per. (Max. Dur.)"})
    feature_counts_tex = feature_counts.style.to_latex(**kwargs)
    return feature_counts_tex
