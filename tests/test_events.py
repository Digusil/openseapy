import unittest

import numpy as np

from snaa.signals import SingleSignal


class SpontaneousActivityEvent(unittest.TestCase):
    def test_eveltdataframe_creation(self):
        raise RuntimeError("todo: tests!")
        event_df = EventDataFrame()

        self.assertEqual(len(event_df.signal_dict), 0)
        self.assertEqual(len(event_df.data), 0)

        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        event_df.add_signal(test)

        self.assertIn(test.name, event_df.signal_dict)
        self.assertEqual(len(event_df.signal_dict), 1)

        event_df.data = event_df.data.append({'test': 1}, ignore_index=True)

        self.assertEqual(len(event_df.data), 1)
        self.assertEqual(list(event_df.data.test), [1.0])

    def test_saving_loading(self):
        raise RuntimeError("todo: tests!")
        event_df = EventDataFrame()

        t = np.linspace(0, 5, 10)
        y = t ** 2

        test = SingleSignal(t=t, y=y)

        event_df.add_signal(test)

        event_df.data = event_df.data.append({'test1': 1}, ignore_index=True)
        event_df.data = event_df.data.append({'test2': 2}, ignore_index=True)
        event_df.data = event_df.data.append({'test3': 3}, ignore_index=True)

        event_df.save('_data/test_eventdataframe.h5')

        event_df2 = EventDataFrame.load('_data/test_eventdataframe.h5')

        self.assertEqual(str(event_df.data), str(event_df2.data))
        self.assertEqual(list(event_df.signal_dict.keys()), list(event_df2.signal_dict.keys()))
        self.assertEqual([signal.get_hash() for signal in list(event_df.signal_dict.values())],
                         [signal.get_hash() for signal in list(event_df2.signal_dict.values())])

        
if __name__ == '__main__':
    unittest.main()