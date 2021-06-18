import numpy as np
import pandas as pd

from .signals import SingleSignal


class SNAADataset:
    def __init__(self, hdf_file, step=1, smoother=None):
        """
        Database class for SNAA. To create a new dataset use:
            SNAADataset.new(hdf_file)

        Parameters
        ----------
        hdf_file: str
            file / path string to the HDF5 database
        step: int, optional
            Step size for data return. Default is 1.
        smoother: Smoother or None
            Smoother for smoothing the data before applying the step. If None, the data will be not smoothed. Default
            is None.
        """
        self._file = hdf_file
        self._step = step
        self._smoother = smoother

        self._load_meta_data()

    @property
    def file(self):
        """
        Returns
        -------
        file / path string to the HDF5 database
        """
        return self._file

    @property
    def step(self):
        """
        Returns
        -------
        step size for data return
        """
        return self._step

    @property
    def smoother(self):
        """
        Returns
        -------
        Smoother for smoothing the data before applying the step. If None, the data will be not smoothed.
        """
        return self._smoother

    @property
    def trace_df(self):
        """
        Returns
        -------
        trace register
        """
        return self._trace_register

    @property
    def primary_name_df(self):
        """
        Returns
        -------
        primary name register
        """
        return self._primary_register

    def _load_meta_data(self):
        """
        Load meta data (primary and trace register).
        """
        self._primary_register = pd.read_hdf(self.file, key='data_registration')
        self._trace_register = pd.read_hdf(self.file, key='trace_registration')

    @staticmethod
    def _trace_name_generator(primary_name, trace_id):
        """
        Generate trace name.

        Parameters
        ----------
        primary_name: str
            primary name of the trace
        trace_id: int
            id of the trace

        Returns
        -------
        trace name: str
        """
        return primary_name + '/' + "t{:03d}".format(trace_id)

    def _get_data(self, trace_name):
        """
        Access data by trace name.

        Parameters
        ----------
        trace_name: str
            name of the trace

        Returns
        -------
        trace data: ndarray
        """
        data = pd.read_hdf(self.file, key='series/' + trace_name)

        if self.smoother is not None:
            data = pd.Series(self.smoother.smooth(data), index=data.index)

        return data[::self.step]

    def __getitem__(self, item):
        """
        emulate dictionary behavior

        Parameters
        ----------
        item: trace name

        Returns
        -------
        trace data: SingleSignal
        """
        trace = self._get_data(item)

        return SingleSignal(t=np.array(trace.index), y=trace.values, name=item)

    def __missing__(self, key):
        raise ValueError("{:s} is not a trace in dataset.".format(key))

    def __len__(self):
        return len(self.trace_df)

    def __iter__(self):
        for key in self.trace_df.index:
            yield key

    def __reversed__(self):
        for key in self.trace_df.index[::-1]:
            yield key

    def __contains__(self, item):
        return item in self.trace_df.index

    def keys(self):
        for key in self:
            yield key

    def values(self):
        for key in self:
            yield self[key]

    def commit(self):
        """
        Commit changes in the registration to file.
        """
        self.primary_name_df.to_hdf(self.file, key='data_registration')
        self.trace_df.to_hdf(self.file, key='trace_registration')

    def register_primary_name(self, name, commit=True, **kwargs):
        """
        Register new primary name.

        Parameters
        ----------
        name: str
            Primary name
        commit: bool, optional
            If True, the changes will be committed. Default is True.
        kwargs:
            Additional attributes that will be saved in the primary name data base.
        """
        data_dict = kwargs
        new_df_row = pd.DataFrame(data_dict, index=[name])

        self._primary_register = self.primary_name_df.append(new_df_row)

        if commit:
            self.commit()

    def add_trace(self, trace_id, primary_name, series, **kwargs):
        """
        Add new trace to dataset.

        Parameters
        ----------
        trace_id: int
            ID of the trace
        primary_name: str
            Primary name of the trace. If the primary name is not registered, the function registers the primary name
            without any attributes.
        series: Series
            Pandas series that will be registered and saved as trace.
        kwargs:
            Additional attributes that will be saved in the trace data base.
        """
        data_dict = kwargs
        data_dict.update({'primary_name': primary_name})

        series_name = self._trace_name_generator(primary_name, trace_id)

        new_df_row = pd.DataFrame(data_dict, index=[series_name])

        if primary_name not in self.primary_name_df:
            # todo: warning, wild primary name registration
            self.register_primary_name(primary_name)

        self._trace_register = self.trace_df.append(new_df_row)

        series.to_hdf(self.file, key='series/' + series_name)

        self.commit()

    @classmethod
    def build(cls, *args, **kwargs):
        container = cls(*args, **kwargs)

        return container

    @classmethod
    def new(cls, hdf_file, additional_primary_attributes: list = None, additional_trace_attributes: list = None,
            **kwargs):
        """
        Create new dataset.

        Parameters
        ----------
        hdf_file: str
            file / path of the new dataset
        additional_primary_attributes: list or None, optional
            Additional attributes for the primary name database. If None, no attributes will be added. Default is None.
        additional_trace_attributes: list or None, optional
            Additional attributes for the trace database. If None, no attributes will be added. Default is None.
        step: int, optional
            Step size for data return. Default is 1.
        smoother: Smoother or None
            Smoother for smoothing the data before applying the step. If None, the data will not be smoothed. Default
            is None.

        Returns
        -------
        dataset object: SNAADataset
        """

        if additional_primary_attributes is None:
            additional_primary_attributes = []

        if additional_trace_attributes is None:
            additional_trace_attributes = []

        primary_registration = pd.DataFrame([], columns=[] + additional_primary_attributes)
        trace_registration = pd.DataFrame([], columns=['primary_name'] + additional_trace_attributes)

        primary_registration.to_hdf(hdf_file, key='data_registration')
        trace_registration.to_hdf(hdf_file, key='trace_registration')

        return cls(hdf_file, **kwargs)
