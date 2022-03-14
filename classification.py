# TODO: import libraries
import pandas as pd

df = pd.read_csv("Dry_Bean_Dataset.csv")

#TODO: check for null values in dataset. if that is the case, average values in
#row and use result to fill null cells
# print(df.isnull().sum()) #No need to work on null vals

#TODO: perform feature scaling on the data (note: not necessary if we do
#decision trees unless we're doing dimensionality reduction)

#TODO: split data into training and testing (use cross validation?)

#TODO: determine if we need to use all of the features, or if omitting any
#leads to higher accuracy

#TODO: train the model (SVM? DT? MLP? Multiple models?)

#TODO: Hyperparameter tuning with validation/learning curves and/or gridsearch?

#TODO: use the model to make predictions

#TODO: Obtain performance metrics (runtime, accuracy, precision, recall, f1, etc.)
#for each model, make comparisons
