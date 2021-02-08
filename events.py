import numpy as np
from cached_property import cached_property
from scipy.optimize import minimize
import pandas as pd

from snaa.core import CoreEvent, CoreSingleSignal, CoreEventList, CoreEventDataFrame
from snaa.event_utils import analyse_capacitor_behavior, search_breaks
from snaa.hdf5_format import save_event_to_hdf5, load_event_from_hdf5, save_eventlist_to_hdf5, load_eventlist_from_hdf5, \
    save_eventdataframe_to_hdf5, load_eventdataframe_from_hdf5
from snaa.utils import Smoother, integral_trapz


class Event(CoreEvent):
    def __init__(self, data: CoreSingleSignal = None, t_start: float = None, t_end: float = None, t_reference: float = None, **kwargs):
        super(Event, self).__init__(data, t_start, t_end, t_reference, **kwargs)

    def save(self, filepath, overwrite=True):
        save_event_to_hdf5(self, filepath, overwrite)

    @classmethod
    def load(cls, filepath: str):
        return load_event_from_hdf5(filepath, use_class=cls)


class EventList(CoreEventList):
    def __init__(self, *args, **kwargs):
        super(EventList, self).__init__(*args, **kwargs)

    def save(self, filepath, overwrite=True):
        save_eventlist_to_hdf5(self, filepath, overwrite)

    @classmethod
    def load(cls, filepath: str):
        return load_eventlist_from_hdf5(filepath, use_class=cls)


    def search_breaks(self, *args, **kwargs):
        for event in search_breaks(*args, **kwargs):
            self.append(event)


class EventDataFrame(CoreEventDataFrame):
    def __init__(self, *args, **kwargs):
        super(EventDataFrame, self).__init__(*args, **kwargs)

    def _simple_analysis(self, event_numbers=None, neg_smoother: Smoother = Smoother(window_len=31, window='hann')):
        if len(self.signal_dict) < 1:
            raise RuntimeError('To do a quick check, signals have to add to the EventDataframe-object!')

        data_dict = {
            'start_t': [],
            'start_y': [],
            'inflections': [],
            'slope': [],
            'slope_lt': [],
            'slope_ly': [],
            'peak_lt': [],
            'peak_ly': [],
            'min_lt': [],
            'min_ly':[],
            'signal_name': []
        }

        for signal_name in self.signal_dict:
            signal = self.signal_dict[signal_name]
            smoothed_signal = signal.to_smoothed_signal(smoother=neg_smoother)

            maximum_mask = np.logical_and(np.abs(smoothed_signal.sign_change_dydt) != 0, smoothed_signal.d2ydt2 < 0)
            minimum_mask = np.logical_and(np.abs(smoothed_signal.sign_change_dydt) != 0, smoothed_signal.d2ydt2 > 0)
            inflection_mask = np.logical_and(smoothed_signal.sign_change_d2ydt2 > 0, smoothed_signal.dydt < 0)

            position_correction = []

            for local_y, local_dydt_sign_change, local_d2ydt2 in zip(smoothed_signal.y,
                                                                     smoothed_signal.sign_change_dydt,
                                                                     smoothed_signal.d2ydt2):
                if len(position_correction) == 0:
                    position_correction.append(local_y)
                elif local_dydt_sign_change < 0:
                    position_correction.append(local_y)
                else:
                    position_correction.append(position_correction[-1])

            position_correction = np.array(position_correction)

            if event_numbers is None:
                position_correction_delta = np.diff(position_correction) != 0
                event_numbers = np.cumsum(np.concatenate(([0, ], position_correction_delta)))

            peak_assumption = []

            for local_y, local_dydt_sign_change, local_d2ydt2 in zip(smoothed_signal.y[::-1],
                                                                     smoothed_signal.sign_change_dydt[::-1],
                                                                     smoothed_signal.d2ydt2[::-1]):
                if len(peak_assumption) == 0:
                    peak_assumption.append(local_y)
                elif local_dydt_sign_change > 0:
                    peak_assumption.append(local_y)
                else:
                    peak_assumption.append(peak_assumption[-1])

            peak_assumption = np.array(peak_assumption[::-1])

            ycorr = smoothed_signal.y - position_correction
            peak_assumption_corr = peak_assumption - position_correction

            for event in range(int(np.nanmax(event_numbers))+1):
                event_mask = event_numbers == event
                try:
                    event_pos = np.where(event_mask)[0][0]
                except IndexError:
                    event_pos = np.NaN

                evaluation_mask = np.logical_and(event_mask, inflection_mask)

                data_dict['signal_name'].append(signal_name)

                if np.any(evaluation_mask) and np.any(np.logical_and(event_mask, minimum_mask)):
                    evaluation_pos = np.argmin(smoothed_signal.dydt[evaluation_mask])

                    try:
                        start_pos = np.where(np.logical_and(event_mask, maximum_mask))[0][0]
                    except IndexError:
                        start_pos = np.where(maximum_mask[:np.where(event_mask)[0][0]])[0][-1]

                    start_t = smoothed_signal.t[start_pos]
                    start_y = smoothed_signal.y[start_pos]

                    data_dict['start_t'].append(start_t)
                    data_dict['start_y'].append(start_y)

                    data_dict['peak_ly'].append(peak_assumption_corr[evaluation_mask][evaluation_pos])
                    data_dict['peak_lt'].append(
                        smoothed_signal.t[
                            np.where(np.logical_and(event_mask, minimum_mask))[0][-1]
                        ] - start_t
                    )

                    data_dict['inflections'].append(np.sum(evaluation_mask))

                    data_dict['slope'].append(np.min(smoothed_signal.dydt[evaluation_mask]))
                    data_dict['slope_lt'].append(
                        smoothed_signal.t[np.where(evaluation_mask)[0][evaluation_pos]] - start_t
                    )
                    data_dict['slope_ly'].append(ycorr[evaluation_mask][evaluation_pos])

                    min_pos = np.argmin(signal.y[event_mask])
                    global_min_t = signal.t[min_pos+event_pos]
                    global_min_y = signal.y[min_pos+event_pos]

                    data_dict['min_lt'].append(global_min_t - start_t)
                    data_dict['min_ly'].append(global_min_y - start_y)

                else:
                    data_dict['start_t'].append(np.NaN)
                    data_dict['start_y'].append(np.NaN)
                    data_dict['inflections'].append(np.NaN)
                    data_dict['slope'].append(np.NaN)
                    data_dict['slope_lt'].append(np.NaN)
                    data_dict['slope_ly'].append(np.NaN)
                    data_dict['peak_lt'].append(np.NaN)
                    data_dict['peak_ly'].append(np.NaN)
                    data_dict['min_lt'].append(np.NaN)
                    data_dict['min_ly'].append(np.NaN)

        return pd.DataFrame.from_dict(data_dict)

    def check_search_settings(
            self,
            neg_threshold: float,
            min_peak_threshold: float = 3,
            neg_smoother: Smoother = Smoother(window_len=31, window='hann'),
            **kwargs
    ):

        event_df = self._simple_analysis(neg_smoother=neg_smoother)

        slope_mask = event_df['slope'] <= neg_threshold
        peak_mask = np.abs(event_df['peak_ly']) >= min_peak_threshold

        number_filtered_events = np.sum(np.logical_and(slope_mask, peak_mask))
        number_slope_filtered_events = np.sum(slope_mask)
        number_peak_filtered_events = np.sum(peak_mask)
        number_all_events = len(event_df)

        filtered_events = event_df.loc[np.logical_and(slope_mask, peak_mask)]

        filtered_events['approx_time_20_80'] = 0.6 * filtered_events.peak_ly / filtered_events.slope

        return \
            number_filtered_events/number_all_events, \
            number_slope_filtered_events/number_all_events, \
            number_peak_filtered_events/number_all_events, \
            filtered_events

    def check_event_mask(
            self,
            event_numbers,
            neg_smoother: Smoother = Smoother(window_len=31, window='hann'), **kwargs
    ):
        event_df = self._simple_analysis(event_numbers=event_numbers, neg_smoother=neg_smoother)

        event_df['approx_time_20_80'] = 0.6 * event_df.peak_ly / event_df.slope

        return event_df

    def search_breaks(self, signal, *args, **kwargs):
        self.add_signal(signal, signal.name)

        for event in search_breaks(signal, *args, **kwargs):
            self.data = self.data.append(event, ignore_index=True)

    def export_event(self, event_id, event_type: type = Event):
        data = self.data.iloc[event_id]

        signal_name = data.signal_name

        event = event_type(
            self.signal_dict[signal_name],
            t_start=data.zero_grad_start_time + data.reference_time,
            t_end=data.zero_grad_end_time + data.reference_time,
            t_reference=data.reference_time,
            y_reference=data.reference_value
        )

        event.start_time = data.start_time
        event.start_value = data.start_value

        event.end_time = data.end_time
        event.end_value = data.end_value

        for key in data.index:
            if getattr(event, key, None) is None:
                event[key] = data[key]

        return event

    def export_event_list(self, event_type: type = Event):
        event_list = EventList()

        for event_id in self.data.index:
            event_list.add_event(self.export_event(event_id, event_type=event_type))

        return event_list

    def save(self, filepath, overwrite=True):
        save_eventdataframe_to_hdf5(self, filepath, overwrite)

    @classmethod
    def load(cls, filepath: str):
        return load_eventdataframe_from_hdf5(filepath, use_class=cls)


class SpontaneousActivityEvent(Event):
    # todo: detecting and analyzing multi peak events

    def __init__(self, *args, **kwargs):
        super(SpontaneousActivityEvent, self).__init__(*args, **kwargs)

        self.register_cached_property('peak_time')
        self.register_cached_property('peak_value')
        self.register_cached_property('pre_peak_time')
        self.register_cached_property('pre_peak_value')
        self.register_cached_property('integral')
        self.register_cached_property('mean_start_slope')

        self._event_data += ['mean_start_slope', 'peak_value', 'peak_time', 'pre_peak_time', 'pre_peak_value']

    @cached_property
    def peak_time(self):
        peak_id = np.argmin(self.y_local)
        return self.t_local[peak_id]

    @cached_property
    def peak_value(self):
        return np.min(self.y_local)

    @cached_property
    def pre_peak_time(self):
        mask = self.t_local <= self.peak_time
        peak_id = np.argmax(self.y_local[mask])
        return self.t_local[peak_id]

    @cached_property
    def pre_peak_value(self):
        mask = self.t_local <= self.peak_time
        return np.max(self.y_local[mask])

    def _capacitor_hypothesis(self, t_local, ymax, tau):
        return ymax - (ymax - self.simplified_peak_end_value) * np.exp(-(t_local - self.simplified_peak_end_time) / tau)

    def approximate_capacitor_behavior(self, cutoff: float = 0.3, iterations: int = 5, smoother: Smoother = Smoother(31), **kwargs):
        self.del_cache()
        ymax, tau, alpha = analyse_capacitor_behavior(self, cutoff=cutoff, iterations=iterations, **kwargs)

        t = self.t_local[self.t_local >= self.peak_time]
        if len(t) > 3:
            if len(t) <= smoother.window_len:
                smoother.window_len = len(t) - 1

            y = smoother.smooth(self.y_local[self.t_local >= self.peak_time])

            loss = lambda par: np.mean((self._capacitor_hypothesis(t, *par) - y) ** 2)

            self.simple_cap_ymax = ymax
            self.simple_cap_tau = tau
            self.simple_cap_alpha = alpha
            self.simple_cap_loss = loss((ymax, tau))
        else:
            self.simple_cap_ymax = np.NaN
            self.simple_cap_tau = np.NaN
            self.simple_cap_alpha = np.NaN
            self.simple_cap_loss = np.NaN

    def capacitor_time_series(self, t_local, type='simple'):
        ymax = self[type+'_cap_ymax']
        tau = self[type+'_cap_tau']

        if ymax is np.NaN or tau is np.NaN:
            return np.NaN
        else:
            return self._capacitor_hypothesis(t_local, ymax, tau)

    def fit_capacitor_behavior(self, smoother: Smoother = Smoother(11, signal_smoothing=False), **kwargs):
        self.del_cache()
        #ymax = self.simple_cap_ymax if self.simple_cap_ymax is not np.NaN else np.max(self.y)
        ymax = 10
        #tau = self.simple_cap_tau if self.simple_cap_tau is not np.NaN else 0
        tau = 1e5

        t = self.t_local[self.t_local >= self.simplified_peak_end_time]
        y = smoother.smooth(self.y_local[self.t_local >= self.simplified_peak_end_time])

        loss = lambda par: np.nanmean((self._capacitor_hypothesis(t, *par) - y) ** 2)

        res = minimize(loss, x0=[ymax, tau], method='Nelder-Mead', options={'max_iter': 10e3})

        if res.status != 0:
            tau = -1e-3
            ymin = self.peak_value - 10

            tmp_res = minimize(loss, x0=[ymin, tau], method='Nelder-Mead', options={'max_iter': 10e3})

            if tmp_res.fun < res.fun:
                res = tmp_res

        self.fitted_cap_ymax = res.x[0]
        self.fitted_cap_tau = res.x[1]
        self.fitted_cap_loss = res.fun
        self.fitted_cap_status = res.status

    @cached_property
    def mean_start_slope(self):
        return (self.peak_value - self.start_value) / (self.peak_time - self.start_time)


def extend_spontaneous_activity_values(event_df):
    event_df.data['pre_peak_time'] = np.repeat(np.NaN, len(event_df.data))
    event_df.data['pre_peak_value'] = np.repeat(np.NaN, len(event_df.data))
    event_df.data['fitted_cap_ymax'] = np.repeat(np.NaN, len(event_df.data))
    event_df.data['fitted_cap_tau'] = np.repeat(np.NaN, len(event_df.data))
    event_df.data['fitted_cap_loss'] = np.repeat(np.NaN, len(event_df.data))
    event_df.data['fitted_cap_status'] = np.repeat(np.NaN, len(event_df.data))
    for event_id in range(len(event_df.data)):
        test_event = event_df.export_event(event_id, event_type=SpontaneousActivityEvent)
        event_df.data['pre_peak_time'].iloc[event_id] = test_event.pre_peak_time
        event_df.data['pre_peak_value'].iloc[event_id] = test_event.pre_peak_value

        test_event.fit_capacitor_behavior()

        event_df.data['fitted_cap_ymax'].iloc[event_id] = test_event.fitted_cap_ymax
        event_df.data['fitted_cap_tau'].iloc[event_id] = test_event.fitted_cap_tau
        event_df.data['fitted_cap_loss'].iloc[event_id] = test_event.fitted_cap_loss
        event_df.data['fitted_cap_status'].iloc[event_id] = test_event.fitted_cap_status