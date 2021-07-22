import unittest

import os

import numpy as np
from scipy.io import savemat
import pandas as pd
from snaa.dataset import SNAADataset
from snaa.loader import HekaMatLoader, collect_data

from .utils import TestCaseWithTemporaryFolder

# todo: unittest for CSV loader classes


class TestHekaMatLoader(TestCaseWithTemporaryFolder):
    def __init__(self, *args, **kwargs):
        super(TestHekaMatLoader, self).__init__(*args, **kwargs)

        self.data = {
            'trace1': np.hstack((np.linspace(0, 9, 10, endpoint=True)[:, None], np.random.rand(10, 1))),
            'trace2': np.hstack((np.linspace(10, 19, 10, endpoint=True)[:, None], np.random.rand(10, 1))),
        }

        self.sources_dict = {
            self.folder('data1.mat'): {
                'sample_time': 1 / 1,
                'type': 'test a',
            },
            self.folder('data2.mat'): {
                'sample_time': 1 / 1,
                'type': 'test b',
            },
            self.folder('data3.mat'): {
                'sample_time': 1 / 1,
                'type': 'test c',
            }
        }

        for file_name in self.sources_dict:
            savemat(file_name, self.data)

    def test_load_mat_file_without_time_row(self):
        loader = HekaMatLoader(sample_rate=1, amplify=1)

        test_data = list(loader(next(iter(self.sources_dict))))

        true_data = None
        for trace in self.data:
            if true_data is None:
                true_data = self.data[trace]
            else:
                true_data = np.hstack((true_data, self.data[trace]))

        np.testing.assert_allclose(true_data, np.array(test_data).T)

    def test_load_mat_file_with_time_row(self):
        loader = HekaMatLoader(amplify=1, time_row=0)

        test_data = list(loader(next(iter(self.sources_dict))))

        true_data = None
        for trace in self.data:
            if true_data is None:
                true_data = self.data[trace][:, 1:]
            else:
                true_data = np.hstack((true_data, self.data[trace][:, 1:]))

        np.testing.assert_allclose(true_data, np.array(test_data).T)

    def test_load_mat_file_check_sample_rate(self):
        loader = HekaMatLoader(sample_rate=1, amplify=1, time_row=0)

        test_data = list(loader(next(iter(self.sources_dict))))

        loader = HekaMatLoader(sample_rate=2, amplify=1, time_row=0)

        with self.assertRaises(AssertionError):
            test_data = list(loader(next(iter(self.sources_dict))))

    def test_collect_data(self):
        loader = HekaMatLoader(sample_rate=1, amplify=1, time_row=0)

        test = collect_data(loader, self.sources_dict, self.folder('test_collect.h5'))

        self.assertIsInstance(test, SNAADataset)

        true_dict = {
            'filename': {},
            'folder': {},
            'sample_time': {},
            'type': {}
        }

        for path in self.sources_dict:
            folder, file_name = os.path.split(path)
            primary_name, ext = os.path.splitext(file_name)

            true_dict['filename'].update({primary_name: file_name})
            true_dict['folder'].update({primary_name: folder})
            true_dict['sample_time'].update({primary_name: self.sources_dict[path]['sample_time']})
            true_dict['type'].update({primary_name: self.sources_dict[path]['type']})

        true_df = pd.DataFrame.from_dict(true_dict)

        pd.util.testing.assert_frame_equal(test.primary_name_df, true_df)

        self.assertEqual(6, len(test.trace_df))

        test = collect_data(loader, self.sources_dict, self.folder('test_collect.h5'), append=True)

        self.assertEqual(12, len(test.trace_df))


if __name__ == '__main__':
    unittest.main()
