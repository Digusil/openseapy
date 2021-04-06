**SNAA** is a python package for detection spontanous events in time series of patchclam signals.

## Installation
Currently, the package is only available on github. To run the **SNAA** package, the [**eventsearch**](https://github.com/digusil/eventsearch) is required. This pack
```shell
git clone https://github.com/digusil/eventsearch
cd eventsearch && pyhton setup.py
```

After installing the **eventsearch** package, you can install the **SNAA** package.
```shell
git clone https://github.com/digusil/snaa
cd eventsearch && pyhton setup.py
```

### Testing
The package has a unittest for the core functions.
```shell
cd ./test && python -m unittest
```

## Acknowledgement
This software was developted on the [institute for process mashinary](https://www.ipat.tf.fau.eu) in cooperation with the [institute for animal physiology](https://www.tierphys.nat.fau.de). 

## License
[Apache License 2.0](LICENSE.txt)
