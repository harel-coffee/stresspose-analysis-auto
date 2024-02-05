from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from empkins_io.sensors.motion_capture.motion_capture_formats import mvnx
from empkins_io.utils._types import path_t, str_t


def _build_data_path(base_path: path_t, subject_id: str, condition: str) -> Path:
    base_path = Path(base_path)
    path = base_path.joinpath(f"{subject_id}/{condition}")
    assert path.exists()
    return path


def _load_tsst_mocap_data(
    base_path: path_t, subject_id: str, condition: str, *, verbose: bool = True
) -> (pd.DataFrame, datetime):
    data_path = _build_data_path(
        base_path.joinpath("data_per_subject"),
        subject_id=subject_id,
        condition=condition,
    )

    mocap_path = data_path.joinpath("mocap/processed")
    mocap_file = mocap_path.joinpath(f"{subject_id}_{condition}-TEST.mvnx")
    if not mocap_file.exists():
        # look for gzip file
        mocap_file = mocap_file.with_suffix(".mvnx.gz")

    if not mocap_file.exists():
        raise FileNotFoundError(f"File '{mocap_file}' not found!")

    mvnx_data = mvnx.MvnxData(mocap_file, verbose=verbose)

    return mvnx_data.data, mvnx_data.start


def _load_gait_mocap_data(
    base_path: path_t,
    subject_id: str,
    condition: str,
    test: str,
    trial: int,
    speed: str,
) -> (pd.DataFrame, datetime):
    data_path = _build_data_path(
        base_path.joinpath("data_per_subject"),
        subject_id=subject_id,
        condition=condition,
    )

    mocap_path = data_path.joinpath("mocap/processed")

    if test == "TUG" or trial == 0:
        mocap_file = mocap_path.joinpath(f"{subject_id}_{condition}-{test}{trial}.mvnx")
    else:
        mocap_file = mocap_path.joinpath(f"{subject_id}_{condition}-{test}{trial}_{speed}.mvnx")

    if not mocap_file.exists():
        mocap_file = mocap_file.with_suffix(".mvnx.gz")

    if not mocap_file.exists():
        raise FileNotFoundError(f"File '{mocap_file}' not found!")

    mvnx_data = mvnx.MvnxData(mocap_file)

    return mvnx_data.data


def _get_times_for_mocap(
    timelog: pd.DataFrame,
    start_time: datetime,
    phase: Optional[str_t] = "total",
) -> pd.DataFrame:
    if phase == "total":
        timelog = timelog.drop(columns="prep", level="phase")
        timelog = timelog.loc[:, [("talk", "start"), ("math", "end")]]
        timelog = timelog.rename({"talk": "total", "math": "total"}, level="phase", axis=1)
    else:
        if isinstance(phase, str):
            phase = [phase]
        timelog = timelog.loc[:, phase]

    timelog = (timelog - start_time).apply(lambda x: x.dt.total_seconds())
    timelog = timelog.T["time"].unstack("start_end")
    return timelog
