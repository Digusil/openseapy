import numpy as np
import pandas as pd
from scipy import signal as scsig, stats
from .utils import ECDF

from eventsearch.core import CoreEvent, CoreSingleSignal, CoreEventList, CoreEventDataFrame

from eventsearch.saving import save_event_to_hdf5, load_event_from_hdf5, save_eventlist_to_hdf5, load_eventlist_from_hdf5, \
    save_eventdataframe_to_hdf5, load_eventdataframe_from_hdf5


class EventDataFrame(CoreEventDataFrame):
    def __init__(self, *args, **kwargs):  #
        """
        event dataframe class
        The instances of this class holds the signals and a pandas dataframe with the event data.
        """
        super(EventDataFrame, self).__init__(*args, **kwargs)

    def _threshold_analysis(self, threshold, window_length):
        """
        Search events only by local extreme values.

        Parameters
        ----------
        threshold: float

        Returns
        -------
        event dataframe: DataFrame
        """
        if len(self.signal_dict) < 1:
            raise RuntimeError('To do a quick check, signals have to add to the EventDataframe-object!')

        data_dict = {
            'peak_t': [],
            'peak_y': [],
            'signal_name': []
        }

        for signal_name in self.signal_dict:
            signal = self.signal_dict[signal_name]

            peaks = threshold_based_analysis(signal, threshold, window_length)

            if len(peaks) > 0:
                for peak in peaks:
                    data_dict['signal_name'].append(signal_name)
                    data_dict['peak_y'].append(signal.y[peak])
                    data_dict['peak_t'].append(signal.t[peak]
                    )
            else:
                data_dict['signal_name'].append(signal_name)
                data_dict['peak_t'].append(np.NaN)
                data_dict['peak_y'].append(np.NaN)

        return pd.DataFrame.from_dict(data_dict)

    def search(self, threshold, window_length):
        self.data = self._threshold_analysis(threshold, window_length)

    def save(self, filepath, overwrite=True):
        """
        Save object as hdf.

        Parameters
        ----------
        filepath: str
            name / path of the file
        overwrite: bool, optional
            should an existing file be overwritten? Default is True.
        """
        save_eventdataframe_to_hdf5(self, filepath, overwrite)

    @classmethod
    def load(cls, filepath: str):
        """
        Load object from hdf.

        Parameters
        ----------
        filepath: str
            name / path of the file

        Returns
        -------
        loaded event dataframe: EventDataFrame
        """
        return load_eventdataframe_from_hdf5(filepath, use_class=cls)


def find_baseline(signal, window_length):
    baseline = []
    for start_id in range(len(signal) - window_length):
        window = signal[start_id:start_id + window_length]

        quantille = ECDF(window).eval(0.33)

        baseline.append(quantille)

        front_buffer = int(np.ceil(window_length / 2))
        back_buffer = len(signal) - len(baseline) - front_buffer

    return np.array(front_buffer * [np.NaN, ] + baseline + back_buffer * [np.NaN, ])


def threshold_based_analysis(signal, threshold, window_length, butter_freqs=[100, 2e3]):
    fs = 1 / np.median(np.diff(signal.t))

    b, a = scsig.butter(3, np.divide(butter_freqs, fs), 'bandpass', analog=False)

    filtered_signal = scsig.lfilter(b, a, signal.y)

    baseline = find_baseline(filtered_signal, window_length)

    filtered_signal -= baseline

    std_approx = 1.4826 * stats.median_abs_deviation(filtered_signal[~np.isnan(filtered_signal)])

    trigger = np.where(np.diff(1.0 * (filtered_signal < - threshold * std_approx)) > 0)[0], \
              np.where(np.diff(1.0 * (filtered_signal < - threshold * std_approx)) < 0)[0]

    return [t1 + np.argmin(signal.y[t1:t2]) for t1, t2 in zip(*trigger)]