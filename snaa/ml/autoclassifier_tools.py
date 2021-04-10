import numpy as np
from eventsearch.utils import Smoother, assign_elements


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
    convertet lables array: ndarray
    """
    return np.array(list(map(lambda x: code_dict[x] if not np.isnan(x) else 0, data)))


def analyse_autoclassifier(data, max_length=250, min_length=30, noise_level=0.25, debug_mode=False):
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
    code_fwd_dict = {
        2: 0,
        4: 1,
        3: 2,
        0: 3,
        6: 4,
        5: 5,
        1: 6
    }

    code_rev_dict = {
        2: 0,
        4: 6,
        3: 5,
        0: 4,
        6: 3,
        5: 2,
        1: 1
    }

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
