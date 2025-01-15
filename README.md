# FairBoosting Regression

This notebook demonstrates the use of **FairBoosting**, specifically **FairBoosting regression**, along with different implementation approaches. 
<br />
The core regressor is based on the academic paper [FairBoost: Boosting supervised learning for learning on multiple sensitive features](https://www.sciencedirect.com/science/article/pii/S0950705123007499).

## Overview

FairBoosting extends the traditional AdaBoost Regressor by incorporating fairness-aware adjustments to ensure balanced performance across protected groups defined by sensitive attributes. The key difference between implementations lies in how the model calculates weights and penalizes poorly classified instances, with the aim of maintaining fairness for multiple sensitive features.

## Features

- **FairBoost Regressor**: An extension of AdaBoost that integrates fairness constraints.
- **Multiple Sensitive Features**: Handles datasets with one or more sensitive attributes.
- **Customizable**: Designed to be applied to any dataset, with flexibility in choosing sensitive variables.
- **Boston Dataset**: Used for evaluation, though the implementation is adaptable to other datasets.

## Usage

To apply this regressor, simply integrate it with your own dataset by specifying sensitive features. The notebook includes detailed code for implementing and tuning the regressor for your needs.
