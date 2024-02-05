"""Hyperparameters for pilot study."""
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_model_dict() -> Dict[str, Dict[str, Any]]:
    """Get model dictionary for pilot study.

    Returns
    -------
    dict
        Model dictionary for pilot study
    """
    return {
        "remove_var": {
            "VarianceThreshold": VarianceThreshold(),
        },
        "scaler": {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
        },
        "reduce_dim": {
            "SelectKBest": SelectKBest(),
            "RFE": RFE(SVC(kernel="linear", C=1)),
        },
        "clf": {
            "GaussianNB": GaussianNB(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "SVC": SVC(probability=True),
            "MLPClassifier": MLPClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
        },
    }


def get_hyper_para_dict(num_subjects: int) -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter dictionary for pilot study.

    Parameters
    ----------
    num_subjects : int
        Number of subjects (dertermines max number of features to select)

    Returns
    -------
    dict
        Hyperparameter dictionary for pilot study
    """
    num_features = list(np.arange(2, num_subjects, 2))
    return {
        "StandardScaler": None,
        "MinMaxScaler": None,
        "VarianceThreshold": {"threshold": [0.0]},
        "SelectKBest": {"k": num_features},
        "RFE": {"n_features_to_select": num_features},
        "GaussianNB": None,
        "KNeighborsClassifier": {"n_neighbors": np.arange(1, num_subjects, 2), "weights": ["uniform", "distance"]},
        "DecisionTreeClassifier": {
            "criterion": ["gini", "entropy"],
            "max_depth": np.arange(1, 20, 2),
            "min_samples_leaf": np.arange(0.1, 0.5, 0.1),
            "min_samples_split": np.arange(0.1, 0.8, 0.1),
            "max_features": [*list(np.arange(0.1, 0.6, 0.1)), "sqrt", "log2", None],
        },
        "SVC": [
            {
                "kernel": ["linear"],
                "C": np.logspace(start=-2, stop=4, num=7),  # 0.01 - 10000
            },
            {
                "kernel": ["rbf"],
                "C": np.logspace(start=-2, stop=4, num=7),  # 0.01 - 10000
                "gamma": np.logspace(start=-4, stop=3, num=8),  # 0.0001 - 1000
            },
            {
                "kernel": ["poly"],
                "C": np.logspace(start=-2, stop=4, num=7),  # 0.01 - 10000
                "degree": np.arange(2, 6),
            },
        ],
        "MLPClassifier": {
            "hidden_layer_sizes": [
                (1,),
                (1, 1),
                (1, 1, 1),
                (2,),
                (2, 2),
                (2, 2, 2),
                (5,),
                (5, 5),
                (5, 5, 5),
            ],
            "max_iter": [
                5000,
            ],
            "activation": ["identity", "tanh", "relu"],
            "solver": ["lbfgs", "adam"],
            "alpha": np.logspace(start=-2, stop=2, num=5),  # 0.01 - 100
        },
        "AdaBoostClassifier": {
            "estimator": [DecisionTreeClassifier(max_depth=1), SVC(kernel="linear", C=1, probability=True)],
            "n_estimators": np.arange(10, 500, 20),
            "learning_rate": list(np.arange(0.01, 0.1, 0.01)) + list(np.arange(0.1, 1.6, 0.1)),
        },
        "RandomForestClassifier": {
            "bootstrap": [True],
            "criterion": ["entropy"],
            "max_depth": [*list(np.arange(4, 50, 4)), None],
            "max_features": [*list(np.arange(0.1, 0.5, 0.1)), "sqrt"],
            "min_samples_leaf": np.arange(0.05, 0.25, 0.05),
            "min_samples_split": np.arange(0.1, 0.5, 0.1),
            "min_weight_fraction_leaf": np.arange(0.1, 0.5, 0.1),
            "max_leaf_nodes": np.arange(2, 20, 2),
            "min_impurity_decrease": np.arange(0, 0.1, 0.01),
            "n_estimators": np.arange(10, 400, 10),
            "ccp_alpha": np.arange(0, 0.1, 0.01),
        },
    }


def get_hyper_search_dict() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter search strategy for pilot study.

    Returns
    -------
    dict
        Hyperparameter search strategy for pilot study
    """
    # use randomized-search for random forest classifier, use grid-search (the default) for all other estimators
    return {"RandomForestClassifier": {"search_method": "random", "n_iter": 40000}}
