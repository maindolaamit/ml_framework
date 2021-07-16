from setuptools import setup, find_packages

import os
import io
from pathlib import Path

from setuptools import setup, find_packages

# Package meta-data.
__name__ = 'ml_framework'
__description__ = 'My framework cum Library for ML and DL.'
__url__ = 'https://github.com/me/myproject'
__email__ = 'maindola.amit@gmail.com'
__author__ = 'Amit Maindola'
__requires_python__ = '>=3.6.0'
__version__ = '0.1.0'
__license__ = 'MIT'

here = Path('__file__').resolve().parent


def get_long_description():
    # Import the README and use it as the long-description.
    # Note: this will only work if 'README.md' is present in your MANIFEST.in file!
    try:
        with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
            long_description = '\n' + f.read()
    except FileNotFoundError:
        long_description = "Unable to read README.md"

    return long_description


def find_requirements():
    with open(os.path.join(here, 'requirements.txt'), 'r') as f:
        return f.read().splitlines()


setup(
    name=__name__,
    version=__version__,
    packages=find_packages(),
    license=__license__,
    author=__author__,
    author_email=__email__,
    description=__description__,
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url='https://github.com/maindolaamit/mllib',
    install_requires=find_requirements(),
)
