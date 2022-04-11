# Applied Machine Learning I - Final Project

Vensan Cabardo, Likhitha Devineni, Camika Leiva

## Package Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/), or a package manager of your choice to install sklearn.

```bash
pip install sklearn
```

## Overview
The purpose of this project is to evaluate which machine learning technique would best be applied to classify species of dry beans. Once a technique is determined to be the best fit for this problem, a solution is provided in the form of a user interface for a user to conveniently receive predictions for their data.

This project contains multiple files:

1. classification.py - Implements multiple sklearn classifiers and obtains certain performance metrics on them (accuracy, precision, recall, f1, and runtime)
2. validation_curves.py - Used to assist in hyperparameter tuning for the machine learning models that are implemented in classification.py
3. classification_tool.py - The actual implementation of our solution to the project. Provides a user interface for users to enter data about a bean and returns a prediction as to what type of bean they entered.

## Dataset
The dataset used for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset):

## Running the Project
#### Run the following command to view performance metrics and analyses of various machine learning algorithms on this dataset:
- Linear SVM
- Kernel SVM
- Decision Tree
- RandomForestClassifier

```bash
python3 classification.py
```

#### Run the following command to view the validation curves generated for the above machine learning models and analyses of various machine learning algorithms on this dataset:

```bash
python3 validation_curves.py
```

#### Run the following command to view the interface that we provide as a solution for this problem:

```bash
python3 classification_tool.py
```
