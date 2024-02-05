"""Module with helper functions to work with the StressPose dataset."""
import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from empkins_io.sensors.motion_capture.perception_neuron import PerceptionNeuronDataset
from empkins_io.utils._types import path_t


def build_data_path(base_path: path_t, subject_id: str, condition: str) -> Path:
    """Build the path to the data for a given subject and condition.

    Parameters
    ----------
    base_path : :class:`pathlib.Path` or str
        Path to the base directory of the dataset.
    subject_id : str
        subject ID
    condition : str
        condition

    Returns
    -------
    :class:`pathlib.Path`
        path to the folder containing the data for the given subject and condition

    """
    base_path = Path(base_path)
    path = base_path.joinpath(f"{subject_id}/{condition}")
    assert path.exists()
    return path


def load_mocap_data(base_path: path_t, subject_id: str, condition: str) -> pd.DataFrame:
    """Load the mocap data for a given subject and condition.

    Parameters
    ----------
    base_path : :class:`pathlib.Path` or str
        Path to the base directory of the dataset.
    subject_id : str
        subject ID
    condition : str
        condition

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame containing the mocap data for the given subject and condition

    """
    data_path = build_data_path(base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition)

    mocap_path = data_path.joinpath("mocap/filtered")
    mocap_data = PerceptionNeuronDataset.from_folder(mocap_path)
    return mocap_data.data_as_df(index="time")


def get_times_for_mocap(
    base_path: path_t, sampling_rate: float, subject_id: str, condition: str, phase: Optional[str] = "total"
) -> Sequence[float]:
    """Get the start and end times for a given phase in mocap time.

    Parameters
    ----------
    base_path : :class:`pathlib.Path` or str
        Path to the base directory of the dataset.
    sampling_rate : float
        sampling rate of the mocap data
    subject_id : str
        subject ID
    condition : str
        condition
    phase : str, optional
        phase to get the times for or "total" to get the total time of the condition. Default: "total"

    Returns
    -------
    list of float
        start and end time of the given phase in mocap time

    """
    data_path = build_data_path(base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition)
    time_file = data_path.joinpath(f"{subject_id}_times_{condition}.json")
    time_json = json.load(time_file.open(encoding="utf-8"))
    clap_mocap_frame = time_json["mocap"]["clap_frame"]
    clap_video_sec = time_json["video"]["clap_sec"]
    times_video = list(time_json["video"][phase].values())
    # subtract clap time to get relative start/end time
    times_relative = [vid_time - clap_video_sec for vid_time in times_video]
    # convert mocap clap time to seconds and add to start/end times
    times_mocap = [clap_mocap_frame / sampling_rate + time_rel for time_rel in times_relative]
    return times_mocap
