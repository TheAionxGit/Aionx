# Machine Learning and Deep Learning for time series forecasting

![MIT License](https://img.shields.io/badge/license-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

This repository contains codes, resources and models for time series forecasting and analysis using Machine Learning and Deep Learning
 
## Models

- Density Hemisphere Neural Network (DensityHNN):
Implements the model proposed in the [Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4627773) : From Reactive to Proactive Volatility Modeling with Hemisphere Neural Networks.

 DensityHNN is a deep Learning algorithm designed to produce Density forecasts on time-dependent data by using an ensemble of deep neural networks. The proposed architecture is the following:

<div align="center">
    <img src="https://github.com/TheAionxGit/Aionx/blob/main/images/DensityHNN_archi.png" alt="DensityHNN Architecture" width="600"/>.
</div>

The network has two independent hemispheres: one estimating the conditional mean (yellow) and one estimating the conditional volatility (blue). Both hemispheres share a common block (red) at the entrance of the network, which performs a non-linear transformation of covariates before sending them to the two hemispheres.

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

2. Install with pip

    ```bash
    pip install aionx
    ```

3. (TODO) explore the example notebooks in the [Link to Tutorial Notebook](examples) directory to get started.

## Dependencies

- [NumPy](https://numpy.org/) : The fundamental package for scientific computing in Python.
- [Pandas](https://pandas.pydata.org/) : An open-source data analysis and manipulation library.
- [TensorFlow](https://www.tensorflow.org/) : An open-source deep learning framework.
- [Scikit-Learn](https://scikit-learn.org/stable/) : An open-source machine learning framework.
      
