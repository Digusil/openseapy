from collections.abc import Iterable
from copy import copy

import numpy as np
from scipy import signal
import pandas as pd
from cached_property import cached_property

from snaa.dataset import SNAADataset as CoreDataset
from snaa.utils import Smoother

import matplotlib.pyplot as plt

import tensorflow as tf


class SNAADataset(CoreDataset):
    def __init__(
            self,
            hdf_file,
            feature_length=11,
            target_length=11,
            shift=0,
            feature_sample_length=None,
            target_sample_length=None,
            **kwargs
    ):
        """
        extended database class for SNAA machine learning

        Parameters
        ----------
        hdf_file: str
            file / path string to the HDF5 database
        feature_length: int
            feature length for machine learning
        target_length: int
            target length for machine learning
        shift: int, optional
            Feature und target data can be shifted. Default is 0.
        feature_sample_length: int or None, optional
            For custom feature length, the data will be resampled by scipy.signal.resample_poly. If None, the data wont
            be resampled. Default is None.
        target_sample_length: int or None, optional
            For custom target length, the data will be resampled by scipy.signal.resample_poly. If None, the data wont
            be resampled. Default is None.
        step: int, optional
            Step size for data return. Default is 1.
        smoother: Smoother or None
            Smoother for smoothing the data before applying the step. If None, the data will be not smoothed. Default
            is None.
        """
        self._position_len = -1

        self._feature_length = feature_length
        self._target_length = target_length
        self._shift = shift

        if feature_sample_length is None:
            self._feature_sample_length = self._feature_length
        else:
            self._feature_sample_length = feature_sample_length

        if target_sample_length is None:
            self._target_sample_length = self._target_length
        else:
            self._target_sample_length = target_sample_length

        super(SNAADataset, self).__init__(hdf_file, **kwargs)

    @property
    def feature_length(self):
        """
        Returns
        -------
        feature length: int
        """
        return self._feature_length

    @property
    def target_length(self):
        """
        Returns
        -------
        target length: int
        """
        return self._target_length

    @property
    def feature_sample_length(self):
        """
        Returns
        -------
        resampled feature length: int
        """
        return self._feature_sample_length

    @property
    def target_sample_length(self):
        """
        Returns
        -------
        resampled target length: int
        """
        return self._target_sample_length

    @property
    def shift(self):
        """
        Returns
        -------
        target - feature shift: int
        """
        return self._shift

    def _calc_max_number(self, trace_name):
        """
        Calculate number of sample windows in trace.

        Parameters
        ----------
        trace_name: str

        Returns
        -------
        number of samples: int
        """
        return len(self._get_data(trace_name)) - np.max(
            [self.feature_sample_length, self.shift + self.target_sample_length])

    @cached_property
    def position_dict(self):
        """
        Generate position dictionary for traces.

        Returns
        -------
        position dictionary: dict
        """
        counter = 0

        position_dict = {}

        for trace in self.trace_df.index:
            n = self._calc_max_number(trace)
            position_dict.update({trace: [counter, counter + n]})

            counter += n

        self._position_len = counter

        return position_dict

    @property
    def position_len(self):
        """
        Returns
        -------
        number of possible samples over all traces: int
        """
        if self._position_len < 0:
            self.position_dict

        return self._position_len

    def _reverse_position_dict(self, sample_id):
        """
        Reverse look up for samples to get corresponding trace.

        Parameters
        ----------
        sample_id: int
            ID of the sample.

        Returns
        -------
        trace name: str
        """
        for trace in self.position_dict:
            if sample_id < self.position_dict[trace][1]:
                break

        return trace, sample_id - self.position_dict[trace][0]

    def _random_positions(self, sample_number):
        """
        Generate random sample list.

        Parameters
        ----------
        sample_number: int
            number of samples

        Returns
        -------
        list of sample ids: list
        """
        return np.random.choice(self.position_len, sample_number, replace=False)

    def sequence_generator(self, sample_number=None, scaler=lambda x, y: [x - np.mean(x), y - np.mean(x)]):
        """
        Generator function for tfdata.

        Parameters
        ----------
        sample_number: int
            Number of samples.
        scaler: callable, optional
            Scaling for feature and target (return as tuple). Default is lambda x, y: [x-np.mean(x), y-np.mean(x)].
        """
        if isinstance(sample_number, list) or isinstance(sample_number, Iterable):
            position_list = sample_number
        elif sample_number is not None and sample_number < self.position_len:
            position_list = self._random_positions(sample_number)
        else:
            position_list = np.arange(self.position_len)

        position_list = np.sort(position_list)

        loaded_trace = ''

        for sample_id in position_list:
            trace_name, idn = self._reverse_position_dict(sample_id)

            if trace_name != loaded_trace:
                data = self._get_data(trace_name)
                loaded_trace = trace_name

                smoother = Smoother(10 * np.max([self.feature_sample_length, self.target_sample_length]))

                baseline = smoother.smooth(data)

                data -= baseline

                offset = data.min()
                data -= offset

                scale = data.max()
                data /= scale

            features = copy(data.iloc[idn:idn + self.feature_sample_length].values)
            # feature_offset = np.mean(features)
            # features -= feature_offset
            # targets /= np.std(targets)

            targets = copy(data.iloc[idn + self.shift:idn + self.shift + self.target_sample_length].values)
            # target_offset = np.mean(targets)
            # targets -= feature_offset
            # features /= np.std(features)

            features, targets = scaler(features, targets)

            if self.feature_length != len(features):
                features = signal.resample_poly(
                    features,
                    self.feature_length, self.feature_sample_length,
                    padtype='line'
                )
            if self.target_length != len(targets):
                targets = signal.resample_poly(
                    targets,
                    self.target_length, self.target_sample_length
                    , padtype='line'
                )

            n_in = len(features)
            features = features.reshape((n_in, 1))

            n_out = len(targets)
            targets = targets.reshape((n_out, 1))

            yield (tf.convert_to_tensor(features, dtype=tf.float32),
                   tf.convert_to_tensor(targets, dtype=tf.float32)
                   )

            # yield (features, targets, (offset, scale, feature_offset, target_offset))

    @classmethod
    def build(cls, *args, **kwargs):
        container = cls(*args, **kwargs)
        container.position_dict

        return container
