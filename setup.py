from setuptools import setup, find_packages

setup(
    name='aion',
    version='1.0.0',
    author='Mikael Frenette',
    author_email = '<mik.frenette@gmail.com>'
    description='Library for designing Machine Learning models for time series forecast',
    packages=find_packages(),
    py_modules=['aion'],
    install_requires=[
        "numpy",
        "pandas==2.0.0",
        "tensorflow",
        "scikit-learn",
    ],
)
