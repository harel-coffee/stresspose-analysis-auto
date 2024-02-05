from functools import cached_property, lru_cache
from itertools import product
from typing import Optional, Sequence, Dict, Union

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.utils._types import path_t, str_t

from stresspose_analysis.datasets.mainstudy._base_dataset import MainStudyBaseDataset
from stresspose_analysis.datasets.mainstudy._helper import _load_tsst_mocap_data, _get_times_for_mocap

_cached_load_mocap_data = lru_cache(maxsize=4)(_load_tsst_mocap_data)


class MainStudyDataset(MainStudyBaseDataset):
    """Dataset class representation for the Main Study dataset."""

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        *,
        exclude_complete_subjects_if_error: bool = True,
        exclude_without_mocap: bool = True,
        use_cache: bool = True,
        verbose: bool = True,
    ):
        """Dataset class representation for the Main Study dataset.

        This class is a :class:`tpcp.Dataset` representation that provides a unified interface for accessing the data of
        the data from the Main Study. It provides access to the questionnaire data, saliva data,
        and motion capture data.

        Parameters
        ----------
        base_path : :class:`pathlib.Path` or str
            path to the base directory of the Main Study dataset
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

        """
        self.verbose = verbose

        super().__init__(
            base_path=base_path,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
            use_cache=use_cache,
            exclude_complete_subjects_if_error=exclude_complete_subjects_if_error,
            exclude_without_mocap=exclude_without_mocap,
        )

    @cached_property
    def mocap_data(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

        subject_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        data, start = self._get_mocap_data(subject_id, condition, verbose=self.verbose)

        times = _get_times_for_mocap(self.timelog_test, start, phase="total")
        times = times.loc["total"]
        data_total = data.loc[times["start"] : times["end"]]

        return data_total

    def _get_mocap_data(self, subject_id: str, condition: str, *, verbose: bool = True) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_mocap_data(self.base_path, subject_id, condition, verbose=verbose)
        return _load_tsst_mocap_data(self.base_path, subject_id, condition, verbose=verbose)


class MainStudyDatasetPerPhase(MainStudyDataset):
    """Dataset class representation for the Main Study dataset, with access to motion capture data per phase."""

    PHASES = ("prep", "talk", "math")

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        *,
        exclude_complete_subjects_if_error: bool = True,
        exclude_without_mocap: bool = True,
        use_cache: bool = True,
        verbose: bool = True,
    ):
        """Dataset class representation for the Main Study dataset.

        This class is a :class:`tpcp.Dataset` representation that provides a unified interface for accessing the data of
        the data from the Main Study. It provides access to the questionnaire data, saliva data,
        and motion capture data.

        Parameters
        ----------
        base_path : :class:`pathlib.Path` or str
            path to the base directory of the Main Study dataset
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

        """
        super().__init__(
            base_path=base_path,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
            use_cache=use_cache,
            exclude_complete_subjects_if_error=exclude_complete_subjects_if_error,
            exclude_without_mocap=exclude_without_mocap,
            verbose=verbose,
        )

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        index = list(product(subject_ids, self.CONDITIONS, self.PHASES))

        index_cols = ["subject", "condition", "phase"]
        index = pd.DataFrame(index, columns=index_cols)
        index = index.set_index(index_cols)
        index = index.drop(index=self.data_to_exclude).reset_index()

        return index

    @cached_property
    def mocap_data(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

        subject_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        phase = self.index["phase"][0] if self.is_single(None) else list(self.index["phase"])

        data_total = self._get_mocap_data_per_phase(subject_id, condition, phase)
        return data_total

    def _get_mocap_data_per_phase(
        self, subject_id: str, condition: str, phase: str_t, *, verbose: bool = True
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        data, start = self._get_mocap_data(subject_id, condition, verbose=verbose)
        timelog = self.timelog_test
        times = _get_times_for_mocap(timelog, start, phase)

        if isinstance(phase, str):
            return data.loc[times.loc[phase, "start"] : times.loc[phase, "end"]]

        data_total = {}
        for ph in phase:
            data_total[ph] = data.loc[times.loc[ph, "start"] : times.loc[ph, "end"]]

        return data_total
