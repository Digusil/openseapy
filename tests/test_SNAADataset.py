import unittest

import os
import numpy as np
import pandas as pd

from snaa.dataset import SNAADataset
from .utils import TestCaseWithTemporaryFolder


class TestSNAADataset(TestCaseWithTemporaryFolder):
    def test_snaadataset_creation(self):
        dataset = SNAADataset.new(self.folder("data.h5"),
                                  additional_primary_attributes=['Ftest1', 'Ftest2'],
                                  additional_trace_attributes=['Ttest3', 'Ttest4']
                                  )

        self.assertIn('data.h5', os.listdir(self.data_folder))

        self.assertEqual(list(dataset.primary_name_df.columns), ['Ftest1', 'Ftest2'])
        self.assertEqual(list(dataset.trace_df.columns), ['primary_name', 'Ttest3', 'Ttest4'])

        del dataset

        dataset = SNAADataset(self.folder("data.h5"), step=2)

        self.assertEqual(list(dataset.primary_name_df.columns), ['Ftest1', 'Ftest2'])
        self.assertEqual(list(dataset.trace_df.columns), ['primary_name', 'Ttest3', 'Ttest4'])

        self.assertEqual(dataset.step, 2)

    def test_snaadataset_register_primary_name(self):
        dataset = SNAADataset.new(self.folder("data.h5"), additional_primary_attributes=['test1', 'test2'])

        dataset.register_primary_name('name1', test1=1, test2='test')

        del dataset

        dataset = SNAADataset(self.folder("data.h5"))

        self.assertEqual(list(dataset.primary_name_df.loc['name1']), [1, 'test'])

    def test_snaadataset_register_trace(self):
        dataset = SNAADataset.new(self.folder("data.h5"), additional_trace_attributes=['test1', 'test2'])

        dataset.register_primary_name('name1')

        series = pd.Series(np.ones(shape=(10,)), index=np.linspace(0, 1, 10))

        dataset.add_trace(0, 'name1', series, test1=1, test2='test')

        del dataset

        dataset = SNAADataset(self.folder("data.h5"))

        self.assertEqual(list(dataset.trace_df.loc['name1/t000']), ['name1', 1, 'test'])
        data = dataset._get_data('name1/t000')

        pd.testing.assert_series_equal(data, series)

    def test_snaadataset_dictionary_behavior(self):
        dataset = SNAADataset.new(self.folder("data.h5"), additional_trace_attributes=['test1', 'test2'])

        dataset.register_primary_name('name1')

        series = pd.Series(np.ones(shape=(10,)), index=np.linspace(0, 1, 10))

        dataset.add_trace(0, 'name1', series, test1=1, test2='test')

        data = dataset['name1/t000']

        pd.testing.assert_series_equal(data.to_series(), series)


if __name__ == '__main__':
    unittest.main()
