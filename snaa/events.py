import numpy as np
from cached_property import cached_property
from scipy.optimize import minimize

from .eventsearch.signals import Smoother
from .eventsearch.events import Event, EventDataFrame
from .eventsearch.event_utils import analyse_capacitor_behavior


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