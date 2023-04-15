"""Module for processing data from the Perception Neuron motion capture suit."""
import sys
from typing import Any, Dict, Sequence

from empkins_io.processing.motion_capture import BvhProcessor, CalcProcessor, CenterOfMassProcessor
from empkins_io.sensors.motion_capture.motion_capture_formats.bvh import BvhData
from empkins_io.sensors.motion_capture.motion_capture_formats.calc import CalcData
from empkins_io.sensors.motion_capture.motion_capture_formats.center_mass import CenterOfMassData


def process_bvh(
    data_dict: Dict[str, BvhData],
    pos_filter_params: Dict[str, Any],
    rot_filter_params: Sequence[Dict[str, Any]],
) -> BvhProcessor:
    """Process BVH data.

    Parameters
    ----------
    data_dict : dict
        dictionary containing BVH data
    pos_filter_params : dict
        parameters for position drift filtering
    rot_filter_params : list of dict
        parameters for rotation drift filtering

    Returns
    -------
    :class:`~empkins_io.processing.motion_capture.BvhProcessor`
        BVH processor

    """
    bvh_proc = BvhProcessor(data_dict["bvh"])
    bvh_data = bvh_proc.filter_position_drift("raw", pos_filter_params)
    bvh_proc.add_data("pos", bvh_data)
    bvh_data = bvh_proc.filter_rotation_drift("pos", rot_filter_params)
    bvh_proc.add_data("rot", bvh_data)
    bvh_data = bvh_proc.global_poses("rot", file=sys.stdout)
    bvh_proc.add_data("global_pose", bvh_data)
    return bvh_proc


def process_calc(
    data_dict: Dict[str, CalcData],
    pos_filter_params: Dict[str, Any],
    rot_filter_params: Sequence[Dict[str, Any]],
) -> CalcProcessor:
    """Process Calc data.

    Parameters
    ----------
    data_dict : dict
        dictionary containing Calc data
    pos_filter_params : dict
        parameters for position drift filtering
    rot_filter_params : list of dict
        parameters for rotation drift filtering

    Returns
    -------
    :class:`~empkins_io.processing.motion_capture.CalcProcessor`
        Calc processor

    """
    calc_proc = CalcProcessor(data_dict["calc"])
    calc_data = calc_proc.filter_position_drift("raw", pos_filter_params)
    calc_proc.add_data("pos", calc_data)
    calc_data = calc_proc.filter_rotation_drift("pos", rot_filter_params)
    calc_proc.add_data("rot", calc_data)
    return calc_proc


def process_center_of_mass(
    data_dict: Dict[str, CenterOfMassData], pos_filter_params: Dict[str, Any]
) -> CenterOfMassProcessor:
    """Process center of mass data.

    Parameters
    ----------
    data_dict : dict
        dictionary containing center of mass data
    pos_filter_params : dict
        parameters for position drift filtering

    Returns
    -------
    :class:`~empkins_io.processing.motion_capture.CenterOfMassProcessor`
        center of mass processor

    """
    com_proc = CenterOfMassProcessor(data_dict["center_mass"])
    com_data = com_proc.filter_position_drift("raw", pos_filter_params)
    com_proc.add_data("pos", com_data)
    return com_proc
