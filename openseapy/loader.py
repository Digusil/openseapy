import os

import numpy as np
import scipy.io as sio

import pandas as pd
from openseapy.dataset import SNADataset


class CoreLoader:
    """
    Core class for data loading.
    """
    def __init__(self, amplify=1e12, time_row=None, sample_rate=None, use_generic_trace_id=True):
        """
        Parameters
        ----------
        amplify: flout, optional
            Factor to amplify signal values. Default is 1e12.
        time_row: int or None, optional
            Row id of the time vector. If None, the time vector will be generated based on the sample rate.
            Default is None.
        sample_rate: float or None, optional
            Sample rate of the data. If None, a time vector cannot be created or proofed. Default is None.
        use_generic_trace_id: bool, optional
            Should the loader generate a generic trace id. Default is True.
        """
        self.amplify = amplify
        self.time_row = time_row
        self.sample_rate = sample_rate
        self.use_generic_trace_id = use_generic_trace_id

    def _load(self, source_file):
        pass

    def __call__(self, source_file):
        return self._load(source_file)


class HekaMatLoader(CoreLoader):
    """
    Loader for Heka mat files.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        amplify: flout, optional
            Factor to amplify signal values. Default is 1e12.
        time_row: int or None, optional
            Row id of the time vector. If None, the time vector will be generated based on the sample rate.
            Default is None.
        sample_rate: float or None, optional
            Sample rate of the data. If None, a time vector cannot be created or proofed. Default is None.
        use_generic_trace_id: bool, optional
            Should the loader generate a generic trace id. Default is True.
        """
        super(HekaMatLoader, self).__init__(*args, **kwargs)

    def _load(self, source_file):
        folder, filename = os.path.split(source_file)
        primary_name = os.path.splitext(filename)[0]

        mat_contents = sio.loadmat(source_file)

        traces_list = list(filter(lambda entry: entry[0] != '_', mat_contents.keys()))

        for trace_id, trace_name in enumerate(traces_list):
            y = mat_contents[trace_name]
            if self.time_row is None:
                if self.sample_rate is None:
                    raise AttributeError('If no time row is defined, sample rate has to be set!')
                time = np.arange(0, len(y) / self.sample_rate, 1 / self.sample_rate)
            else:
                time = mat_contents[trace_name][:, self.time_row]
                fs_file = np.median(1 / np.diff(mat_contents[trace_name][:, self.time_row]))

                if self.sample_rate is not None:
                    assert 0.01 > np.abs(fs_file / self.sample_rate - 1), \
                        "Problem with sample rate: {0:} != {1:}".format(self.sample_rate, fs_file)

            for idr, row in enumerate(y.T):
                if idr == self.time_row:
                    continue

                if self.use_generic_trace_id:
                    name = primary_name + '/t{:03d}'.format(trace_id) + '/r{:03d}'.format(idr)
                else:
                    name = primary_name + '/' + trace_name + '/r{:03d}'.format(idr)

                yield pd.Series(row * self.amplify, index=time, name=name)


class CSVLoader(CoreLoader):
    """
    Loader class for CSV data.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        amplify: flout, optional
            Factor to amplify signal values. Default is 1e12.
        time_row: int or None, optional
            Row id of the time vector. If None, the time vector will be generated based on the sample rate.
            Default is None.
        sample_rate: float or None, optional
            Sample rate of the data. If None, a time vector cannot be created or proofed. Default is None.
        """
        super(CSVLoader, self).__init__(*args, **kwargs)

        if self.use_generic_trace_id is not True:
            raise ValueError('CSVLoader do not support non generic trace ids!')

    def _load(self, source_file, **kwargs):
        folder, filename = os.path.split(source_file)
        primary_name = os.path.splitext(filename)[0]

        y = pd.read_csv(source_file, **kwargs).values

        if self.time_row is None:
            if self.sample_rate is None:
                raise AttributeError('If no time row is defined, sample rate has to be set!')
            time = np.arange(0, len(y) / self.sample_rate, 1 / self.sample_rate)
        else:
            time = y[:, self.time_row]
            fs_file = np.median(1 / np.diff(y[:, self.time_row]))

            if self.sample_rate is not None:
                assert 0.01 > np.abs(fs_file / self.sample_rate - 1), \
                    "Problem with sample rate: {0:} != {1:}".format(self.sample_rate, fs_file)

        for idr, row in enumerate(y.T):
            if idr == self.time_row:
                continue

            name = primary_name + '/t{:03d}'.format(0) + '/r{:03d}'.format(idr)

            yield pd.Series(row * self.amplify, index=time, name=name)


def collect_data(loader, sources_dict, target_file, append=False):
    """
    Collect raw data.

    Parameters
    ----------
    loader: CoreLoader
        loader object
    sources_dict: dictionary
        dictionary with information about the files (key) and additional meta information (value)
    target_file: str
        path to the file to store the data
    append: bool, optional
        If true, the data will be appended, else the file will be overwritten. Default False.

    Returns
    -------
    SNADataset
    """

    config_columns = sources_dict[next(iter(sources_dict))].keys()
    file_registration = pd.DataFrame([], columns=['filename', 'folder', *list(config_columns)])
    trace_registration = pd.DataFrame([], columns=['primary_name'])

    for source_file in sources_dict:
        folder, filename = os.path.split(source_file)
        primary_name = os.path.splitext(filename)[0]

        config = sources_dict[source_file]

        file_registration.loc[primary_name] = [filename, folder, *list(config.values())]

        for series in loader(source_file):
            series_name = series.name
            trace_registration.loc[series_name] = [primary_name]

            series.to_hdf(target_file, key='series/' + series_name)

    if append:
        try:
            previous_file_registration = pd.read_hdf(target_file, key='data_registration')
            previous_trace_registration = pd.read_hdf(target_file, key='trace_registration')

            file_registration = previous_file_registration.append(file_registration)
            trace_registration = previous_trace_registration.append(trace_registration)
        except KeyError:
            pass

    file_registration.to_hdf(target_file, key='data_registration')
    trace_registration.to_hdf(target_file, key='trace_registration')

    return SNADataset(target_file)
