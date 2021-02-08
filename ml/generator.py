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
        return self._feature_length

    @property
    def target_length(self):
        return self._target_length

    @property
    def feature_sample_length(self):
        return self._feature_sample_length

    @property
    def target_sample_length(self):
        return self._target_sample_length

    @property
    def shift(self):
        return self._shift

    def _calc_max_number(self, trace_name):
        return len(self._get_data(trace_name)) \
               - np.max([self.feature_sample_length, self.shift + self.target_sample_length])

    @cached_property
    def position_dict(self):
        counter = 0

        position_dict = {}

        for trace in self.trace_df.index:
            n = self._calc_max_number(trace)
            position_dict.update({trace: [counter, counter+n]})

            counter += n

        self._position_len = counter

        return position_dict

    @property
    def position_len(self):
        if self._position_len < 0:
            self.position_dict

        return self._position_len

    def _reverse_position_dict(self, sample_id):
        for trace in self.position_dict:
            if sample_id < self.position_dict[trace][1]:
                break

        return trace, sample_id - self.position_dict[trace][0]

    def _random_positions(self, sample_number):
        return np.random.choice(self.position_len, sample_number, replace=False)

    def sequence_generator(self, sample_number=None, scaler=lambda x, y: [x-np.mean(x), y-np.mean(x)]):
        """
        old version

        todo:
            - implement n choices to reduce the amount of needed memory
            - loading from pandas hdf file
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
            #feature_offset = np.mean(features)
            #features -= feature_offset
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

            #yield (features, targets, (offset, scale, feature_offset, target_offset))

    @classmethod
    def build(cls, *args, **kwargs):
        container = cls(*args, **kwargs)
        container.position_dict

        return container


if __name__ == '__main__':
    test = SNAADataset('_data/data.h5', target_length=200, feature_sample_length=200)

    print(test.position_dict)
    print(test._reverse_position_dict(249989))
    print(test._reverse_position_dict(test.position_len - 1))

    for features, targets, metadata in list(test.sequence_generator(5)):
        plt.plot(features)
        plt.plot(targets)
        print(metadata)

    plt.show()