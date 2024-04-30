# URL-Based Topic Classification



[![DOI](https://zenodo.org/badge/792693041.svg)](https://zenodo.org/doi/10.5281/zenodo.11093040)



## Introduction

This repository contains the code for simplified experiments based on my seminar paper "Evaluation of Hyperparameter Tuning and Stochastic Gradient Descent for Optimizing URL-Classifying Support Vector Machines" at the University of Passau.
These particular experiments are published as part of the "Data Stewardship" course at the Technical University of Vienna.

## Description

The goal of the URL-based topic classification task is to classify websites into different classes, based solely on the URL.
These experiments evaluate the impact of hyperparameter tuning on the performance of classification algorithms.
Specifically, we evaluate the "alpha" parameter sklearn's "hinge" SGDClassifier, which is a stochastic gradient descent classifier based on a linear support vector machine.
We use two datasets for this task: "DMOZ", which labels each URL with one out of 15 categories (e.g. "Arts", "Business", "Science"), and "PhishStorm", which labels each URL as either "phishing" or "benign".

## Structure

The repository is structured as follows:

- `experiments/`: Contains the experiments for both datasets in Jupyter notebooks. `sgd-hinge-alphasearch-dmoz.ipynb` contains the experiments for the DMOZ dataset, and `sgd-hinge-alphasearch-phishstorm.ipynb` contains the experiments for the PhishStorm dataset.
- `data/`: Contains the datasets used in the experiments.
- `results/charts/`: Contains the charts generated by the experiments.
    - `phishstorm-gridsearch.png`: Grid search results for the PhishStorm dataset showing the impact of the "alpha" parameter on the accuracy.
    - `dmoz-gridsearch.png`: Grid search results for the DMOZ dataset showing the impact of the "alpha" parameter on the accuracy.
    - `dmoz-confusion-matrix.png`: Confusion matrix for the best model on the DMOZ dataset.
- `results/predictions/`: Contains the predictions of the best models for both datasets.
    - `dmoz-predictions.csv`: Predictions of the best model on the DMOZ dataset.
    - `phishstorm-predictions.csv`: Predictions of the best model on the PhishStorm dataset.


## Usage

### Requirements

- Python 3.10 or higher
- Run `pip install -r experiments/requirements.txt` to install the required packages.

### Running the experiments

1. Open the Jupyter notebooks in the `experiments/` directory.
2. Run the cells in the notebooks to reproduce the experiments.
3. The results will be saved in the `results/` directory.

## Datasets attribution

### DMOZ

Source: https://www.kaggle.com/datasets/shawon10/url-classification-dataset-dmoz

### Phishing URL dataset

Tamal, Maruf (2023), “Phishing Detection Dataset”, Mendeley Data, V1, doi: 10.17632/6tm2d6sz7p.1
