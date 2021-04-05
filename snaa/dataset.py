import pandas as pd


class SNAADataset:
    def __init__(self, hdf_file, step=1, smoother=None):
        """
        Database class for SNAA.

        Parameters
        ----------
        hdf_file: str
            file / path string to the HDF5 database
        step: int, optional
            Step size for data return. Default 1.
        smoother: Smoother or None
            Smoother for smoothing the data before applying the step. If None, the data will be not smoothed. Default
            None.
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
        Step size for data return.
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
            Primary name of the trace.
        trace_id: int
            Id of the trace.

        Returns
        -------
        trace name: str
        """
        return primary_name + '/' + "t{:03d}".format(trace_id)

    def _get_data(self, trace_name):
        """
        Acces data by trace name.

        Parameters
        ----------
        trace_name: str
            name of the trace.

        Returns
        -------
        trace data: ndarray
        """
        data = pd.read_hdf(self.file, key='series/' + trace_name)

        if self.smoother is not None:
            data = pd.Series(self.smoother.smooth(data), index=data.index)

        return data[::self.step]

    @classmethod
    def build(cls, *args, **kwargs):
        container = cls(*args, **kwargs)

        return container