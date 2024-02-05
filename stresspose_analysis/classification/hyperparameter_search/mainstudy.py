from typing import Any, Dict, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

_base_model_dict = {
    "remove_var": {
        "VarianceThreshold": VarianceThreshold(),
    },
    "scaler": {},
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
    },
}

_base_hyper_para_dict = {
    "VarianceThreshold": {"threshold": [0.0]},
    "MinMaxScaler": None,
    "StandardScaler": None,
    "PCA": {},
    "SelectKBest": {"k": None},
    "RFE": {"n_features_to_select": None},
    "GaussianNB": None,
    "KNeighborsClassifier": {"n_neighbors": None, "weights": ["uniform", "distance"]},
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
        "solver": ["adam"],
        "alpha": np.logspace(start=-2, stop=2, num=5),  # 0.01 - 100
    },
    "AdaBoostClassifier": {
        "estimator": [DecisionTreeClassifier(max_depth=1), SVC(kernel="linear", C=1, probability=True)],
        "n_estimators": np.arange(10, 500, 20),
        "learning_rate": list(np.arange(0.01, 0.1, 0.01)) + list(np.arange(0.1, 1.6, 0.1)),
    },
}


def get_model_dict(*, scaler: Optional[str] = None, fsel: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    model_dict = _base_model_dict.copy()
    if scaler is None:
        scaler = ["minmax", "standard"]
    if isinstance(scaler, str):
        scaler = [scaler]

    if "minmax" in scaler:
        model_dict["scaler"]["MinMaxScaler"] = MinMaxScaler()
    if "standard" in scaler:
        model_dict["scaler"]["StandardScaler"] = StandardScaler()

    if fsel == "pca":
        model_dict["reduce_dim"] = {}
        model_dict["reduce_dim"]["PCA"] = PCA()

    return model_dict


def get_model_dict_rf() -> Dict[str, Dict[str, Any]]:
    model_dict = {
        "remove_var": {
            "VarianceThreshold": VarianceThreshold(),
        },
        "scaler": {
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
        },
        "reduce_dim": {
            "SelectFromModel": SelectFromModel(RandomForestClassifier(n_estimators=100)),
        },
        "clf": {
            "RandomForestClassifier": RandomForestClassifier(),
        },
    }

    return model_dict


def get_hyper_para_dict_rf(*, num_subjects: int) -> Dict[str, Dict[str, Any]]:
    num_features = list(np.arange(10, num_subjects, 2))
    hyper_para_dict = {
        "VarianceThreshold": {"threshold": [0.0]},
        "MinMaxScaler": None,
        "StandardScaler": None,
        "RFE": {"n_features_to_select": num_features},
        "SelectFromModel": {"threshold": ["mean", "median", "0.5*mean", "0.25*mean", "0.75*mean", "1.25*mean"]},
        "RandomForestClassifier": {
            "bootstrap": [True],
            "criterion": ["entropy"],
            "max_depth": [*list(np.arange(5, 50, 5)), None],
            "max_features": [*list(np.arange(0.1, 0.5, 0.1)), "sqrt"],
            "min_samples_leaf": np.arange(0.05, 0.25, 0.05),
            "min_samples_split": np.arange(0.1, 0.5, 0.1),
            "min_weight_fraction_leaf": np.arange(0.1, 0.5, 0.1),
            "max_leaf_nodes": np.arange(2, 20, 2),
            "min_impurity_decrease": np.arange(0, 0.1, 0.01),
            "n_estimators": np.arange(100, 400, 20),
            "ccp_alpha": np.arange(0, 0.1, 0.01),
        },
    }
    return hyper_para_dict


def get_hyper_para_dict(*, num_subjects: int) -> Dict[str, Dict[str, Any]]:
    num_features = list(np.arange(2, num_subjects, 2))
    hyper_para_dict = _base_hyper_para_dict.copy()

    hyper_para_dict["SelectKBest"]["k"] = num_features
    hyper_para_dict["RFE"]["n_features_to_select"] = num_features
    hyper_para_dict["PCA"]["n_components"] = [0.75, 0.80, 0.85, 0.90] + num_features[::2]

    hyper_para_dict["KNeighborsClassifier"]["n_neighbors"] = np.arange(1, num_subjects, 2)
    return hyper_para_dict


def get_hyper_search_dict() -> Dict[str, Dict[str, Any]]:
    return {}


def get_hyper_search_dict_rf() -> Dict[str, Dict[str, Any]]:
    # use randomized-search for random forest classifier
    return {"RandomForestClassifier": {"search_method": "random", "n_iter": 40000}}
