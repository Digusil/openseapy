from copy import copy

import h5py
import numpy as np
from eventsearch.utils import Smoother, assign_elements

from h5py._hl.base import phil
from h5py._hl import attrs


def arrange_data(data, code_dict):
    """
    Convert ml labels with dictionary.

    Parameters
    ----------
    data: ndarray
        ml label data
    code_dict: dict
        dictionary to convert the labels

    Returns
    -------
    converted labels array: ndarray
    """
    return np.array(list(map(lambda x: code_dict[x] if not np.isnan(x) else 0, data)))


def analyse_autoclassifier(
        data,
        max_length=250,
        min_length=30,
        noise_level=0.25,
        noise_label=2,
        ordered_labels=None,
        debug_mode=False
):
    """
    Analyse label data and extract events.

    Parameters
    ----------
    data: ndarray
        ml label data
    max_length: int, optional
        maximum event length
    min_length: int, optional
        minimum event length
    noise_level: float, optional
        Maximum noise proportion for valid event. Default is 0.25.
    noise_label: int, optional
        Label of the noise. Default 2.
    ordered_labels: list or None, optional
        Forward ordered labels of the signal states. Default is None ([4, 3, 0, 6, 5, 1]).
    debug_mode: bool, optional
        Is True, if additional values will be returned. Default is False.

    Returns
    -------
    ac_events: list
        Start and end position of event as tuple.
    ac_event_numbers: ndarray
        Event mask with ids.

    Additional in debug mode:
    ac_data_fwd_smoothed: ndarray
        forward data
    ac_data_rev_smoothed: ndarray
        reverse data
    ac_event_start: ndarray
        start trigger
    ac_event_end: ndarray
        end trigger
    """
    if ordered_labels is None:
        ordered_labels = [4, 3, 0, 6, 5, 1]

    all_labels = [noise_label, ] + ordered_labels

    assert len(all_labels) == len(np.unique(all_labels)), "Labels have to be unique!"

    code_fwd_dict = {}
    for idl, label in enumerate(all_labels):
        code_fwd_dict.update({label: idl})

    code_rev_dict = {}
    for idl, label in enumerate([noise_label, ] + ordered_labels[::-1]):
        code_rev_dict.update({label: idl})

    ac_data_fwd = arrange_data(data, code_fwd_dict)
    ac_data_rws = arrange_data(data, code_rev_dict)

    ac_smoother = Smoother(window_len=17, window='blackmanharris')

    ac_data_fwd_smoothed = ac_smoother.smooth(ac_data_fwd)
    ac_data_rev_smoothed = ac_smoother.smooth(ac_data_rws)

    # ac_event = ac_smoother.smooth(ac_data_fwd != 0) > 0.9

    ac_event_end = np.array([0, ] + np.diff(ac_data_fwd_smoothed).tolist())
    ac_event_end = ac_event_end < -0.5
    # ac_event_end = ac_smoother.smooth(ac_event_end) >= 0.95

    ac_event_start = np.array([0, ] + np.diff(ac_data_rev_smoothed).tolist())
    ac_event_start = ac_event_start > 0.5
    # ac_event_start = ac_smoother.smooth(ac_event_start) >= 0.95

    ac_event_start_pos = np.where(np.array([0, ] + np.diff(1 * ac_event_start).tolist()) < 0)[0]
    ac_event_end_pos = np.where(np.array(np.diff(1 * ac_event_end).tolist() + [0, ]) > 0)[0]

    ac_events = []
    ac_event_numbers = np.empty(len(data))
    ac_event_numbers.fill(np.NaN)

    last_end = -1
    for ide, pos in enumerate(ac_event_start_pos):
        end_id = np.sum(ac_event_end_pos < pos)

        if end_id > last_end:
            if end_id >= len(ac_event_end_pos):
                ac_events.append([pos, np.NaN])
            else:
                end_pos = ac_event_end_pos[end_id]
                if end_pos - pos <= max_length:
                    noise_check = np.mean(ac_data_fwd[pos:end_pos] == 0)

                    if noise_check <= noise_level and end_pos - pos >= min_length:
                        last_end = end_id
                        ac_events.append([pos, end_pos])
                        ac_event_numbers[pos:end_pos] = ide

    if debug_mode:
        return ac_events, ac_event_numbers, ac_data_fwd_smoothed, ac_data_rev_smoothed, ac_event_start, ac_event_end
    else:
        return ac_events, ac_event_numbers


class MimicHDFFile(h5py.File):
    """
    Tensorflow checks for h5py.File instances. To combine multiple models in one hdf file, the models have to be stored
    in groups not in file. The h5py.File-class is an instances of h5py.Group-class. Thus, h5py.File-objects are extended
    h5py.Group-objects. This class casts a h5py.Group-object as a h5py.File-object and adepts methods:
        - flush method is deactivated:
            Tensorflow handle the h5py.Group-objects like a file and want to flush the data at the end. The model saving
            is only a part of the operations on the current file. Thus, the flush method of this class is deactivated.
        - attrs property is corrected:
            A h5py.File-object writes its attributes always to "/". This is a correct behavior for a h5py.File-object.
            In this case, the object is a group and the attributes have to be stored to the group and not to root.
    """
    def __init__(self, orig_obj):
        """
        Cast the original object to a MimicHDFFile-object.

        Parameters
        ----------
        orig_obj: h5py.Group
        """
        self.__dict__ = copy(orig_obj.__dict__)

    def flush(self):
        """
        Flush method does nothing.
        """
        pass

    @property
    def attrs(self):
        """
        Attributes attached to this group.

        Returns
        -------
        group attributes
        """
        with phil:
            return attrs.AttributeManager(self)