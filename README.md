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

## Acknowledgement
This software was developed on the [institute for process machinery](https://www.ipat.tf.fau.eu) in cooperation with the [institute for animal physiology](https://www.tierphys.nat.fau.de). 

## License
[Apache License 2.0](LICENSE.txt)
