'''
project:    snaa
filename:	version
author:		pi
date:		2020-04-06
version:	

description:

CHANGE LOG:

    - 0.1.0:
        initial version based on wecanalysis

        SingleSignal class for handling signal data
        SmoothedSignal class for smoothing the signal data
        Event class as simple event data container
        EventList class to store a bunch of events
        SpontaneousActivityEvent class as a specialized event class
        saving and loading of Events and EventLists

    - 0.1.1
        better start point based on slope and start value of smoothed signal
        performance improvement through caching in EventList class
        break search improvements
        nadaraya-watson estimator with variance estimator
        implement Estimator class
        implement EventDataFrame class
        dataset class with sequence generator for tensorlfow
        resampling data in sequence generator

    - 0.1.2
        divide code in packages
        add docstrings and untitests
        release on github

    - 0.1.3
        dictionary behavior for SNAADataset
        threshold based detection for comparison

    - 0.1.4
        advanced threshold based detection from https://doi.org/10.1016/S0956-5663(02)00053-2 as comparison

'''
__version__ = '0.1.4a'
