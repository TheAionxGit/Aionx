from setuptools import setup, find_packages

setup(
    name='aionx',
    version='2.0.0',
    author='Mikael Frenette',
    author_email="<mik.frenette@gmail.com>",
    description='Library for designing Machine Learning models for time series forecast',
    packages=find_packages(),
    py_modules=['aionx'],
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow",
        "scikit-learn",
    ],
)
