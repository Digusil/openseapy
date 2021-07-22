import os
import shutil
import unittest

from eventsearch.core_utils import IdentifedObject


class TemporaryFolder(IdentifedObject):
    def __init__(self, **kwargs):
        super(TemporaryFolder, self).__init__(**kwargs)

        self._path = self.__identifier__

    def __call__(self, *args, **kwargs):
        return self._path

    def folder(self, path):
        return self._path + '/' + path

    def __enter__(self):
        os.mkdir(self._path)

        return self

    def __del__(self):
        try:
            shutil.rmtree(self._path)
        except FileNotFoundError:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self._path)


class TestCaseWithTemporaryFolder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCaseWithTemporaryFolder, self).__init__(*args, **kwargs)

        self._folder = TemporaryFolder()
        self._folder.__enter__()

    @property
    def data_folder(self):
        return self._folder()

    def folder(self, path):
        return self._folder.folder(path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._folder

    def __del__(self):
        del self._folder