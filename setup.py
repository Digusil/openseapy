#! /opt/conda/bin/python3
""" General PyPI compliant setup.py configuration of the package """

from setuptools import setup, find_packages

version = {}
with open("openseapy/version.py") as fp:
    exec(fp.read(), version)

__author__ = 'Thomas Pircher'
__version__ = version['__version__']
__copyright__ = '2021, FAU-iPAT'
__license__ = 'Apache-2.0'
__maintainer__ = 'Thomas Pircher'
__email__ = 'thomas.pircher@fau.de'
__status__ = 'Development'

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


def get_readme():
    """
    Method to read the README.rst file

    :return: string containing README.md file
    """
    with open('README.md') as file:
        return file.read()


# ------------------------------------------------------------------------------
#   Call setup method to define this package
# ------------------------------------------------------------------------------
setup(
    name='opensSEApy',
    version=__version__,
    author=__author__,
    author_email=__email__,
    description='A python based analysis of spontaneous excitatory postsynaptic currents (spEPSCs).',
    long_description=get_readme(),
    url='https://github.com/digusil/snaa',
    license=__license__,
    keywords=[
        'openseapy',
        'spontaneous neuron activity',
        'patch clamp analysis',
        'eventsearch',
        'event detection',
        'neuroscience',
        'computational biology'
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    zip_safe=False
)
