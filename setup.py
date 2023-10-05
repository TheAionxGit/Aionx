# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:49:46 2023

@author: User
"""

from setuptools import setup, find_packages

setup(
    name='Prototyping',
    version='0.05',
    author='Mikael Frenette',
    description='Library for designing Machine Learning models for time series forecast',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas==2.0.0",
        "tensorflow==2.10",
        "scikit-learn",
    ],
)
