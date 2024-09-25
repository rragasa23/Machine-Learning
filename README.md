# Gas Holdup Prediction in Bubble Columns Using Random Forest Regression

## Project Overview

This project was undertaken as part of a machine learning class at Virginia Tech. The objective was to predict the gas holdup value in a bubble column using a Random Forest Regression model. Gas holdup is a critical parameter in bubble column reactors, which are widely used in chemical and biochemical processes.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation and Usage](#installation-and-usage)
- [Acknowledgments](#acknowledgments)

## Introduction

Bubble columns are vertical reactors where gas is introduced at the bottom and flows upwards through a liquid medium. The gas holdup, defined as the volume fraction of gas in the column, is a key parameter influencing the reactor's performance. Accurate prediction of gas holdup can enhance the design and operation of these reactors.

## Dataset

The dataset used in this project consists of 4042 data points collected from various bubble column experiments (Hazare et al., 2022). Each data point includes features such as gas flow rate, liquid properties, column dimensions, and operating conditions, along with the corresponding gas holdup value. It is important to note that the majority of the data involves studies using air-water systems.

The source for the data: https://www.sciencedirect.com/science/article/abs/pii/S0263876222002891

## Methodology

1. **Data Preprocessing**: 
    - Using pandas dataframe.
    - Normalizing/Standardizing the features.
    - Splitting the data into training and testing sets.

2. **Model Selection, Training, and Evaluation**:
    - Implementing a Random Forest Regression Model.
    - Training the model using a dataset.
    - Evaluating the model's performance on the test dataset using metrics such as Mean Squared Error (MSE) and R-squared value.

3. **Feature Redundancy Evaluation**:
    - Obtain covariance matrix to address pairwise relationships between each feature.
    - Use PCA explained variance ratio to understand the relative importance of each principal component.
    - Refit model witthout redunant features

4. **Optimize Model**
    - Use grid search method to obtain opimum model and hyperparameters.

5. **Feature Selection**
    - Use feature_importances_ method to generate quantitative weights each feature has on the model.

## Model Training and Evaluation

- **Training**: The Random Forest Regression model was trained on 80% of the dataset.
- **Hyperparameter Tuning**: Performed using Grid Search Cross Validation to optimize the number of trees, max depth, and other parameters.
- **Evaluation**: The model's performance was evaluated on the remaining 20% of the dataset.

## Results

The Random Forest Regression model demonstrated strong predictive performance with the following key metrics:

- **Mean Squared Error (MSE)**: 0.00049
- **R-squared**: 0.956

The model successfully captured the complex relationships between the input features and the gas holdup value, outperforming simpler regression models.

## Conclusion

The project demonstrated that Random Forest Regression is an effective technique for predicting gas holdup in bubble columns. The model provides valuable insights that can aid in the design and optimization of bubble column reactors.

## Installation and Usage

1. **Clone the repository**:
    ```bash
    git clone git@github.com:rragasa23/Machine-Learning.git
    cd gas-holdup-prediction
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the model**:
    ```bash
    python main.py
    ```

4. **Jupyter Notebook**:
    - For detailed analysis and visualization, open the Jupyter Notebook `Final Project.ipynb`.

## Acknowledgments

- This project was completed as part of the Machine Learning course at Virginia Tech.
- Thanks to Professor Hongliang Xin for guidance and support.

## License

