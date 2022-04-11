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

This project is structured as follows:
```bash
.
├── Dry_Bean_Dataset.csv
├── README.md
├── classification.py
└── validation_curves.py
```

## Dataset
The dataset used for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset):

From the data description for this dataset in the UCI Machine Learning Repository:

> "For the classification model, images of 13,611 grains of 7 different registered dry beans were taken with a high-resolution camera. Bean images obtained by computer vision system were subjected to segmentation and feature extraction stages, and a total of 16 features; 12 dimensions and 4 shape forms, were obtained from the grains."

Attribute Information:
1. Area (A): The area of a bean zone and the number of pixels within its boundaries.
2. Perimeter (P): Bean circumference is defined as the length of its border.
3. Major axis length (L): The distance between the ends of the longest line that can be drawn from a bean.
4. Minor axis length (l): The longest line that can be drawn from the bean while standing perpendicular to the main axis.
5. Aspect ratio (K): Defines the relationship between L and l.
6. Eccentricity (Ec): Eccentricity of the ellipse having the same moments as the region.
7. Convex area (C): Number of pixels in the smallest convex polygon that can contain the area of a bean seed.
8. Equivalent diameter (Ed): The diameter of a circle having the same area as a bean seed area.
9. Extent (Ex): The ratio of the pixels in the bounding box to the bean area.
10. Solidity (S): Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.
11. Roundness (R): Calculated with the following formula: (4piA)/(P^2)
12. Compactness (CO): Measures the roundness of an object: Ed/L
13. ShapeFactor1 (SF1)
14. ShapeFactor2 (SF2)
15. ShapeFactor3 (SF3)
16. ShapeFactor4 (SF4)
17. Class (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira)

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
