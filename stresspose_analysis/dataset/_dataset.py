from functools import cached_property, lru_cache
from itertools import product
from typing import Optional, Sequence, Tuple

import pandas as pd
from biopsykit.io import load_long_format_csv, load_questionnaire_data
from biopsykit.utils.dataframe_handling import multi_xs, wide_to_long
from biopsykit.utils.file_handling import get_subject_dirs
from empkins_io.utils._types import path_t
from tpcp import Dataset

from stresspose_analysis.dataset.helper import get_times_for_mocap, load_mocap_data

_cached_load_mocap_data = lru_cache(maxsize=4)(load_mocap_data)


class StressPoseDataset(Dataset):
    """Dataset class representation for the StressPose dataset."""

    SUBJECTS_WITHOUT_MOCAP: Tuple[str] = ("VP_01",)

    base_path: path_t
    exclude_without_mocap: bool
    normalize_mocap_time: bool
    use_cache: bool
    _sampling_rate: float = 1.0 / 0.017
    _sample_times: Tuple[int] = (-20, -1, 0, 10, 20, 45)

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        exclude_without_mocap: Optional[bool] = True,
        normalize_mocap_time: Optional[bool] = True,
        use_cache: Optional[bool] = True,
    ):
        """Dataset class representation for the StressPose dataset.

        This class is a :class:`tpcp.Dataset` representation that provides a unified interface for accessing the data of
        the StressPose dataset. It provides access to the questionnaire data, saliva data, and motion capture data.

        Parameters
        ----------
        base_path : :class:`pathlib.Path` or str
            path to the base directory of the StressPose dataset
        groupby_cols : list of str, optional
            columns to group the data by. Default: ``None``
            **Note:** This parameter is only needed internally when creating subsets of the dataset.
            Leave this as ``None`` when instantiating the dataset.
        subset_index : list of str, optional
            index of the subset to create. Default: ``None``
            **Note:** This parameter is only needed internally when creating subsets of the dataset.
            Leave this as ``None`` when instantiating the dataset.
        exclude_without_mocap : bool, optional
            whether to exclude subjects without motion capture data. Default: ``True``
        normalize_mocap_time : bool, optional
            whether to normalize the motion capture time index, i.e. set the beginning of the cut mocap data to 0.
            Default: ``True``

        """
        # ensure pathlib
        self.base_path = base_path
        self.exclude_without_mocap = exclude_without_mocap
        self.use_cache = use_cache
        self.normalize_mocap_time = normalize_mocap_time
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]
        conditions = ["ftsst", "tsst"]
        if self.exclude_without_mocap:
            for subject_id in self.SUBJECTS_WITHOUT_MOCAP:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)

        index = list(product(conditions, ["talk"]))
        index.append(("tsst", "math"))
        index = [(subject, *i) for subject, i in product(subject_ids, index)]
        index = pd.DataFrame(index, columns=["subject", "condition", "phase"])
        return index

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the motion capture data in Hz.

        Returns
        -------
        float
            sampling rate of the motion capture data in Hz
        """
        return self._sampling_rate

    @property
    def sample_times(self) -> Sequence[int]:
        """Saliva sample times relative to TSST start in minutes.

        Returns
        -------
        list of int
            saliva sample times relative to TSST start in minutes
        """
        return self._sample_times

    @property
    def questionnaire(self):
        """Questionnaire data.

        Returns
        -------
        :class:`pandas.DataFrame`
            questionnaire data
        """
        if self.is_single(["phase"]):
            raise ValueError("questionnaire data can not be accessed for individual phases!")
        if self.is_single(["condition"]):
            raise ValueError("questionnaire data can not be accessed for a single condition!")
        return self._load_questionnaire_data()

    @property
    def condition_first(self) -> pd.DataFrame:
        """Information which condition (TSST or f-TSST) was performed first.

        Returns
        -------
        :class:`pandas.DataFrame`
            overview of which condition (TSST or f-TSST) was performed first

        """
        data = self.questionnaire[["TSST_first"]].replace({True: "TSST first", False: "fTSST first"})
        data.columns = ["condition_first"]
        return data

    @property
    def cortisol_non_responder(self) -> pd.DataFrame:
        """Information whether a subject is a cortisol non-responder.

        A subject is considered a cortisol non-responder if the maximum increase in cortisol levels after TSST is below
        1.5 nmol/l.

        Returns
        -------
        :class:`pandas.DataFrame`
            overview of whether a subject is a cortisol non-responder

        """
        non_responder = self.cortisol_features.xs("tsst", level="condition")
        non_responder = non_responder.xs("max_inc", level="saliva_feature") <= 1.5
        non_responder.columns = ["non_responder"]
        return non_responder

    @property
    def cortisol(self) -> pd.DataFrame:
        """Cortisol data.

        Returns
        -------
        :class:`pandas.DataFrame`
            cortisol data

        """
        return self._load_saliva_data("cortisol")

    @property
    def cortisol_features(self) -> pd.DataFrame:
        """Cortisol features.

        Returns
        -------
        :class:`pandas.DataFrame`
            cortisol features

        """
        return self._load_saliva_features("cortisol")

    @property
    def amylase(self) -> pd.DataFrame:
        """Amylase data.

        Returns
        -------
        :class:`pandas.DataFrame`
            amylase data

        """
        return self._load_saliva_data("amylase")

    @property
    def amylase_features(self) -> pd.DataFrame:
        """Amylase features.

        Returns
        -------
        :class:`pandas.DataFrame`
            amylase features

        """
        return self._load_saliva_features("amylase")

    @cached_property
    def mocap_data(self) -> pd.DataFrame:
        """Motion capture data.

        Returns
        -------
        :class:`pandas.DataFrame`
            motion capture data

        """
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            data = self._get_mocap_data(subject_id, condition)
            phase = self.index["phase"].unique()[0] if self.is_single(None) else "total"
            times = get_times_for_mocap(self.base_path, self.sampling_rate, subject_id, condition, phase)
            data = data.loc[times[0] : times[1]]
            if self.normalize_mocap_time:
                data.index -= data.index[0]
            return data
        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @property
    def panas_diff(self) -> pd.DataFrame:
        """Difference of PANAS questionnaire pre-post (f-)TSST.

        Returns
        -------
        :class:`pandas.DataFrame`
            difference of PANAS questionnaire pre-post (f-)TSST

        """
        panas_data = wide_to_long(self.questionnaire, "PANAS", levels=["subscale", "condition", "time"]).dropna()
        panas_data = panas_data.drop("Total", level="subscale")
        panas_data = panas_data.reindex(["ftsst", "tsst"], level="condition").reindex(["pre", "post"], level="time")
        panas_data = panas_data.unstack("time").diff(axis=1).stack().droplevel(-1)
        return self._apply_indices(panas_data).reorder_levels(
            ["subject", "condition", "condition_first", "non_responder", "subscale"]
        )

    @property
    def stadi_state_diff(self) -> pd.DataFrame:
        """Difference of STADI-State questionnaire pre-post (f-)TSST.

        Returns
        -------
        :class:`pandas.DataFrame`
            difference of STADI-State questionnaire pre-post (f-)TSST

        """
        stadi_data = wide_to_long(self.questionnaire, "STADI_State", levels=["subscale", "condition", "time"]).dropna()
        stadi_data = stadi_data.reindex(["pre", "post"], level="time")
        stadi_data = stadi_data.unstack("time").diff(axis=1).stack().droplevel(-1)
        return self._apply_indices(stadi_data).reorder_levels(
            ["subject", "condition", "condition_first", "non_responder", "subscale"]
        )

    @property
    def pasa(self) -> pd.DataFrame:
        """PASA questionnaire.

        Returns
        -------
        :class:`pandas.DataFrame`
            PASA questionnaire

        """
        pasa_data = wide_to_long(self.questionnaire, "PASA", levels=["subscale"]).dropna()
        return self._apply_indices(pasa_data).reorder_levels(
            ["subject", "condition_first", "non_responder", "subscale"]
        )

    def _apply_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.join(self.condition_first).join(self.cortisol_non_responder)
        data = data.set_index(["condition_first", "non_responder"], append=True)
        return data

    def _get_mocap_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_mocap_data(self.base_path, subject_id, condition)
        return load_mocap_data(self.base_path, subject_id, condition)

    def _load_questionnaire_data(self) -> pd.DataFrame:
        data_path = self.base_path.joinpath("questionnaire_total/processed/empkins_macro_questionnaire_data.csv")
        data = load_questionnaire_data(data_path)
        subject_ids = self.index["subject"].unique()
        return data.loc[subject_ids]

    def _load_saliva_data(self, saliva_type: str) -> pd.DataFrame:
        if self.is_single("phase"):
            raise ValueError(f"{saliva_type} data can not be accessed for individual phases!")
        data_path = self.base_path.joinpath(f"saliva_total/processed/empkins_macro_{saliva_type}.csv")
        data = load_long_format_csv(data_path)

        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return multi_xs(multi_xs(data, subject_ids, level="subject"), conditions, level="condition")

    def _load_saliva_features(self, saliva_type: str) -> pd.DataFrame:
        if self.is_single("phase"):
            raise ValueError(f"{saliva_type} features can not be accessed for individual phases!")
        data_path = self.base_path.joinpath(f"saliva_total/processed/empkins_macro_{saliva_type}_features.csv")
        data = load_long_format_csv(data_path)

        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return multi_xs(multi_xs(data, subject_ids, level="subject"), conditions, level="condition")

    def add_cortisol_index(self, cort_data: pd.DataFrame) -> pd.DataFrame:
        """Add further indices to cortisol data.

        This function adds the indices `condition_first` and `non_responder` to the given cortisol data.

        Parameters
        ----------
        cort_data : :class:`pandas.DataFrame`
            cortisol data

        Returns
        -------
        :class:`pandas.DataFrame`
            cortisol data with additional indices

        """
        index_levels = list(cort_data.index.names)
        new_index_levels = ["condition_first", "non_responder"]
        cort_data = cort_data.join(self.condition_first).join(self.cortisol_non_responder)
        cort_data = cort_data.set_index(new_index_levels, append=True)
        cort_data = cort_data.reorder_levels(index_levels[:-1] + new_index_levels + [index_levels[-1]])

        return cort_data
