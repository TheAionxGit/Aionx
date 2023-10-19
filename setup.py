from setuptools import setup, find_packages

setup(
    name='chronomia',
    version='1.0.0',
    author='Mikael Frenette',
    description='Library for designing Machine Learning models for time series forecast',
    packages=find_packages(),
    py_modules=['chronomia'],
    install_requires=[
        "numpy",
        "pandas==2.0.0",
        "tensorflow",
        "scikit-learn",
    ],
)
