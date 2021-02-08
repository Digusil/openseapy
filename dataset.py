import pandas as pd


class SNAADataset:
    def __init__(self, hdf_file, step=1, smoother=None):
        """
        Database class for SNAA.

        @param hdf_file: path string to the HDF5 database
        @param step: step size for data return (Default = 1)
        @param smoother: Smoother for smoothing the data for applying the step.
        """
        self._file = hdf_file
        self._step = step
        self._smoother = smoother

        self._load_meta_data()

    @property
    def file(self):
        return self._file

    @property
    def step(self):
        return self._step

    @property
    def smoother(self):
        return self._smoother

    @property
    def trace_df(self):
        return self._trace_register

    @property
    def primary_name_df(self):
        return self._primary_register

    def _load_meta_data(self):
        self._primary_register = pd.read_hdf(self.file, key='data_registration')
        self._trace_register = pd.read_hdf(self.file, key='trace_registration')

    @staticmethod
    def _trace_name_generator(primary_name, trace_id):
        return primary_name + '/' + "t{:03d}".format(trace_id)

    def _get_data(self, trace_name):
        data = pd.read_hdf(self.file, key='series/' + trace_name)

        if self.smoother is not None:
            data = pd.Series(self.smoother.smooth(data), index=data.index)

        return data[::self.step]

    @classmethod
    def build(cls, *args, **kwargs):
        container = cls(*args, **kwargs)

        return container