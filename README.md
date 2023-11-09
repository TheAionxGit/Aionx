# Machine Learning and Deep Learning for time series forecasting

![MIT License](https://img.shields.io/badge/license-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

This repository contains codes, resources and models for Time Series forecasting and analysis using Machine Learning and Deep Learning
 
## Models

- Density Hemisphere Neural Network (DensityHNN):
Implements the model proposed in [Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4627773). DensityHNN is a deep Learning algorithm designed to produce Density forecasts on time-dependent data by using an ensemble of deep neural networks. The proposed architecture is the following:

<div align="center">
    <img src="https://github.com/TheAionxGit/Aionx/blob/main/images/DensityHNN_archi.png" alt="DensityHNN Architecture" width="600"/>.
</div>
After estimation, the model is capable of producing conditional forecasts along with uncertainty estimates.

<div align="center">
    <img src="https://github.com/TheAionxGit/Aionx/blob/main/images/GDP_S1.png" alt="GDP S1" width="600"/>.
</div>

A simple usage tutorial for the density hemisphere neural network is available here [example](examples/DensityHNN_tutorial1.ipynb).

## Getting Started

1. Clone this repository:

    ```bash
    git clone https://github.com/TheAionxGit/aionx.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. (TODO) explore the example notebooks in the [Link to Tutorial Notebook](examples/tutorial_notebook.ipynb) directory to get started.

## Usage

This repository provides a simple and flexible framework for time series analysis. You can start with the example notebooks in the `examples/` directory to understand how to use the provided functions and classes.

## Dependencies

- [NumPy](https://numpy.org/) : The fundamental package for scientific computing in Python.
- [Pandas](https://pandas.pydata.org/) : An open-source data analysis and manipulation library.
- [TensorFlow](https://www.tensorflow.org/) : An open-source deep learning framework.
- [Scikit-Learn](https://scikit-learn.org/stable/) : An open-source machine learning framework.
      
```bash
pip install numpy pandas==2.0.0 tensorflow scikit-learn
