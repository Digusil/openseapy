import warnings

import numpy as np
import pandas as pd

from ..events import EventDataFrame as OrigEventDataFrame


class EventDataFrame(OrigEventDataFrame):
    def __init__(self, *args, **kwargs):  #
        """
        event dataframe class
        The instances of this class holds the signals and a pandas dataframe with the event data.
        """
        super(EventDataFrame, self).__init__(*args, **kwargs)

    def _threshold_analysis(self, threshold, slope, bin_width, pp, noise):
        """
        Search events only by local extreme values.

        Parameters
        ----------
        threshold: float
            threshold factor based on the deviation
        slope: float
            minimum slope of events
        bin_width: int
            number of averaged points
        pp: int
            period of peak search
        noise: float
            baseline noise

        Returns
        -------
        event dataframe: DataFrame
        """
        if len(self.signal_dict) < 1:
            raise RuntimeError('To do a quick check, signals have to add to the EventDataframe-object!')

        df_list = []

        for signal_name in self.signal_dict:
            signal = self.signal_dict[signal_name]

            tmp_df = advanced_threshold(signal, threshold, slope, bin_width, pp, noise)
            tmp_df['signal_name'] = signal_name

            df_list.append(tmp_df)

        return pd.concat(df_list, ignore_index=True)

    def threshold_based_search(self, *args, **kwargs):
        """
        Search events based on threshold.

        Parameters
        ----------
        threshold: float
            threshold factor based on the deviation
        slope: float
            minimum slope of events
        bin_width: int
            number of averaged points
        pp: int
            period of peak search
        noise: float
            baseline noise
        """
        warnings.warn("'threshold_based_search' will be removed in the future. Use 'search'!", DeprecationWarning)
        self.search(*args, **kwargs)

    def search(self, threshold, slope, bin_width, pp, noise):
        """
        Search events based on a threshold.

        Parameters
        ----------
        threshold: float
            minimum event amplitude
        slope: float
            minimum slope of events
        bin_width: int
            number of averaged points
        pp: int
            period of peak search
        noise: float
            baseline noise
        """
        self.data = self._threshold_analysis(threshold, slope, bin_width, pp, noise)


def advanced_threshold(signal, threshold, slope, bin_width, pp, noise):
    """
    Threshold based event search based on https://doi.org/10.1016/S0956-5663(02)00053-2

    Parameters
    ----------
    signal: SingleSignal
        signal that will be analyzed
    threshold: float
        minimum event amplitude
    slope: float
        minimum slope of events
    bin_width: int
        number of averaged points
    pp: int
        period of peak search
    noise: float
        baseline noise

    Returns
    -------
    event dataframe: DataFrame
    """
    event_dict = {
        'start_t': [],
        'start_y': [],
        'peak_t': [],
        'peak_y': []
    }

    last_peak_point = -1

    wi = int(threshold / slope * signal.fs)

    i_trigger = -signal.y[wi:] > -signal.y[:-wi] + threshold

    for i in np.where(i_trigger)[0]:
        if i > last_peak_point:
            on_set_point = i

            for j in np.arange(i, i + wi + 1):
                if np.mean(-signal.y[j:j + bin_width]) >= -signal.y[i] + noise:
                    on_set_point = j
                    break

            peak_point = i + wi

            for l in np.arange(on_set_point, on_set_point + pp + 1):
                if l < len(signal):
                    if np.mean(-signal.y[l:l + bin_width]) <= -signal.y[l] - noise:
                        peak_point = l
                        break

            if peak_point != i + wi:
                for m in np.arange(on_set_point, l + 1)[::-1]:
                    if np.mean(-signal.y[m - bin_width:m]) <= -signal.y[l]:
                        peak_point = m
                        break

            on_set_amp = signal.y[on_set_point]
            peak_amp = signal.y[peak_point]

            if -peak_amp + on_set_amp >= threshold:
                last_peak_point = peak_point
                event_dict['start_t'].append(signal.t[on_set_point])
                event_dict['start_y'].append(on_set_amp)
                event_dict['peak_t'].append(signal.t[peak_point])
                event_dict['peak_y'].append(peak_amp)

    return pd.DataFrame.from_dict(event_dict)
