# SNAA

![License](https://img.shields.io/github/license/Digusil/snaa.svg) ![Build status](https://github.com/Digusil/snaa/actions/workflows/python-package.yml/badge.svg?branch=master) ![Version](https://img.shields.io/github/v/release/Digusil/snaa.svg)

**SNAA** is a python package for the detection of spontaneous events in time series of patchclamp signals.

## Installation
Currently, the package is only available on github. To run the **SNAA** package, the [**eventsearch**](https://github.com/digusil/eventsearch) is required.
```shell
git clone https://github.com/digusil/eventsearch
cd eventsearch && pyhton setup.py install
```

If the eventsearch source is cloned into the snaa source, the testing will fail, because the imports in the 'eventesearch/tests/' will not be resolved propperly. To prevent this use separate source folders or delete the 'eventesearch/tests/' folder. 

Recommended folder structure:

    .                   # source folder
    ├── ...
    ├── eventsearch     # eventsearch code
    │   ├── ...
    │   └── tests
    └── snaa            # snaa code
        ├── ... 
        └── tests
        
After installing the **eventsearch** package, you can install the **SNAA** package. 
```shell
git clone https://github.com/digusil/snaa
cd snaa && pyhton setup.py
```

### Testing
The package has a unittest for the core functions. Run the test in the 'snaa/' or 'snaa/tests/' folder.
```shell
python -m unittest
```
Alternative:
```shell
pytest
```

## Using
The data has to be converted to a SingleSignal object to be analyzed. This can be done manually by creating a blank 
SingleSignal object and adding the time and signal data or be loading a single or a group of Heka-mat- or generic 
csv-files. 

#### manually
```python
from snaa.loader import CSVLoader
from snaa.signals import SingleSignal

series = next(CSVLoader(time_row=0)('a_file.csv'))
t = series.index
y = series.values

signal = SingleSignal(t, y, name='test_data')
```

After loading the arrays for time `t` and signal values `y`, the arrays will be given to the SingleSignal initializer. 
In this case, the data will be loaded by the library own loader class for CSV files. It is possible to use alternative 
methods corresponding to your data and needs.  The parameter name is optional. You can set a custom name. The signal 
will be registered in the global __signal_names__ variable. If you try to register an additional signal with the same
name, the program rises a NameError to prevent multiple or misleading signals. If this is not necessary, the 
registration of the signal can be deactivated with the parameter `listed=False`.

#### Loader classes
```python
from snaa.loader import CSVLoader, collect_data

loader = CSVLoader(amplify=1e15, time_row=0, sample_rate=50e3)

sources_dict = {
    'data/file_1.csv': {
        'type': 'cortex cell',
        'comment': ''
    },
    'data/file_2.csv': {
        'type': 'cortex cell',
        'comment': ''
    },
    'data/file_3.csv': {
        'type': 'horizontal cell',
        'comment': 'different cell!'
    },
}

dataset = collect_data(loader, sources_dict, 'data/collected_data.h5')
```

First, the loader function will be created. In this case, the signal values will be amplified by 1e15 instead of the
default 1e12. The time vector will be proofed with both parameter `time_row=0` and `sample_rate=50e3` set. If the sample
rate of the time vector differs from the given value, a AssertionError will be risen. The `sorces_dict` dictionary
defines the location of the data as keys and additional parameter, which will be stored in the dataframe, as values. 

By calling `collect_data(loader, sources_dict, 'data/collected_data.h5')` the function iterates over the dictionary and
extract, register the traces and store the traces in the hdf file *data/collected_data.h5*. Additionally, the function 
returns a SNAADataset object containing of the loaded data. 

### Dataset class
The SNAADataset class can be used to handle and preprocess the data. An additional benefit is the reduction of needed
RAM, because the signal data will be loaded dynamically form the hdf file. The SNAADataset object reacts like a 
dictionary. To access a signal use square brackets and the trace name like it is stored in the `trace_df` dataframe.

For machine learning an extended version of the SNAADataset exist in *ml/generators.py*. This version can generate a 
tfdata instance that can directly be used for training and validation.

### Analyzing 
````python
from snaa.events import EventDataFrame

event_df = EventDataFrame()
event_df.add_signal(signal)

event_df.search(neg_threshold=-7e3, pos_threshold=2e3, min_peak_threshold=5)
event_df.save('data/events.h5')
````

First, an EventDataFrame object have to be created. To analyze signals, the signals have to be added to the 
EventDataFrame object. By calling the `search` method, the added signals will be analyzed with the *slope* algorithm.
The event dataframe can be accessed via die attribute `data`. EventDataFrame objects can be saved. The resulting hdf 
file contains the dataframe with the event data and the signals. Thus, the saved file contains all needed information. 

The *slope* algorithm generates several event values. *_time* marks a relative time position related to the 
*reference_time* value and *_value* marks a relative signal value related to the *reference_value*. Thus, all values are
directly comparable. The calculated values are:

- **zero_grad_start**: previous 0 gradient before the event start
- **start**: start of the event (linearized)
- **peak**: peak of the event
- **end**: end of the event
- **zero_grad_end**: first 0 gradient after the end of the event
- **slope**: maximum slope of the step
- **half_rising**: 50% of the step
- **rising_20**: 20% of the step
- **rising_80**: 80% ot the step
- **simplified_peak_start**: start of the peak for the linearized event shape
- **simplified_peak_start**: end of the peak for the linearized event shape
- **rising_time**: duration of the step
- **recovery_time**: duration of the recovery
- **integral**: area of the event
- **phase_counter**: number of positive inflection points during the step
- **previous_event_time_period**: period between the end of the previous and start of the current event
- **previous_event_reference_period**: period of the reference points to the previous event
- **intersection_problem**: the current event starts before the previous event ends
- **overlapping**: the current peak exists before the previous event ends
- **event_complex**: the current event has intersecting events before or after
- **pre_peak**: local maximum before the peak
- **fitted_camp_ymax**: settle value of the fitted exponential function
- **fitted_camp_tau**: time constant of the fitted exponential function
- **fitted_camp_loss**: loss of the fit
- **fitted_camp_status**: status of the fit corresponding to [scipy minimize result](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult)

The fit of the exponential function can be deactivated by `search(extend=False)`. This reduces the calculation time. 

## Acknowledgement
This software was developed on the [institute for process machinery](https://www.ipat.tf.fau.eu) in cooperation with the [institute for animal physiology](https://www.tierphys.nat.fau.de). 

## License
[Apache License 2.0](LICENSE.txt)
