from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
from biopsykit.classification.model_selection import SklearnPipelinePermuter
from biopsykit.utils.dataframe_handling import add_space_to_camel, snake_to_camel
from empkins_macro.utils._types import str_t

from stresspose_analysis.data_wrangling import rename_motion_features


def flatten_wide_format_column_names(data: pd.DataFrame, col_index_name: Optional[str] = "feature") -> pd.DataFrame:
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
    best_estimator_summary = pipeline_permuter.best_estimator_summary()

    best_pipeline = best_estimator_summary.loc[pipeline].iloc[0]
    fsels = [pipe["reduce_dim"] for pipe in best_pipeline.pipeline]
    num_features = [np.sum(fsel.get_support()) for fsel in fsels]
    num_features = pd.DataFrame(num_features, columns=["num_features"])
    num_features.index.name = "fold"
    return num_features


def feature_counts_to_latex(feature_counts: pd.DataFrame, **kwargs) -> str:
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


def shap_values_per_fold(
    pipeline_permuter: SklearnPipelinePermuter, pipeline: Tuple[str], data: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    best_estimator_summary = pipeline_permuter.best_estimator_summary()
    pipeline_folds = best_estimator_summary.loc[pipeline]["best_estimator"].pipeline

    test_folds = pipeline_permuter.metric_summary().loc[pipeline]["test_indices_folds"]
    test_folds_flat = [idx for sublist in test_folds for idx in sublist]

    shap_per_fold = []
    for fold, pipeline in enumerate(pipeline_folds):
        X_test = data.iloc[list(test_folds[fold]), :]
        estimator = pipeline[-1]

        # Create Tree Explainer object that can calculate shap values
        explainer = shap.TreeExplainer(estimator)
        shap_values_fold = np.array(explainer.shap_values(X_test))
        shap_per_fold.append(shap_values_fold)

    shap_per_fold = [np.transpose(shap_values, [1, 2, 0]) for shap_values in shap_per_fold]
    shap_values = np.concatenate(shap_per_fold)
    shap_values = np.transpose(shap_values, [2, 0, 1])

    return shap_values, data.iloc[test_folds_flat]


def get_shap_feature_importances(
    shap_values: np.ndarray, data: pd.DataFrame, index_levels: Optional[str_t] = None
) -> pd.DataFrame:
    if index_levels is None:
        index_levels = ["subject", "condition"]
    if isinstance(index_levels, str):
        index_levels = [index_levels]
    levels = list(data.index.names)
    for level in index_levels:
        levels.remove(level)

    data_unstack = data.unstack(levels)["data"]

    shap_values = np.mean(np.abs(shap_values), axis=0)
    feature_importance = pd.DataFrame(shap_values, columns=["feature_importance_vals"], index=data_unstack.columns)
    feature_importance = feature_importance.sort_values(by=["feature_importance_vals"], ascending=False)
    return feature_importance
