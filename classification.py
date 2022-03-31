# TODO: import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt                        # inserted for when we begin to plot the results of the classifiers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from timeit import default_timer as timer

# Camika placed these here
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df = pd.read_csv("Dry_Bean_Dataset.csv")

#TODO: check for null values in dataset. if that is the case, average values in
#row and use result to fill null cells
# print(df.isnull().sum()) #No need to work on null vals

X=df[df.columns[0:16]]
y=df[['Class']]

#TODO: split data into training and testing (use cross validation?)
# currently, the test size is 30% of the total data within the dataset,
# we can test more of the dataset later on, if needed.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

#TODO: perform feature scaling on the data (note: not necessary if we do
#decision trees unless we're doing dimensionality reduction)
scaler = StandardScaler()
x_train_std = scaler.fit_transform(X_train)
x_test_std = scaler.fit_transform(X_test)

#TODO: determine if we need to use all of the features, or if omitting any
#leads to higher accuracy

#TODO: train the model (SVM? DT? MLP? Multiple models?)
svm = SVC(kernel='linear', C=0.01, random_state=1)

start = timer()
svm.fit(x_train_std, np.ravel(y_train))
end = timer()

y_pred = svm.predict(x_test_std)

print("-------")
print(f"Accuracy (Linear SVM): {accuracy_score(y_pred, y_test)}")
print(f"Precision (Linear  SVM): {precision_score(y_pred, y_test, average='weighted')}")
print(f"f1 (Linear SVM): {f1_score(y_pred, y_test, average='weighted')}")
print(f"Recall (Linear SVM): {recall_score(y_pred, y_test, average='weighted')}")
print(f"Runtime (Linear SVM): {end - start}")

svm = SVC(kernel='rbf', C=0.01, random_state=1)

start = timer()
svm.fit(x_train_std, np.ravel(y_train))
end = timer()

y_pred = svm.predict(x_test_std)

print("-------")
print(f"Accuracy (Kernel SVM): {accuracy_score(y_pred, y_test)}")
print(f"Precision (Kernel SVM): {precision_score(y_pred, y_test, average='weighted')}")
print(f"f1 (Kernel SVM): {f1_score(y_pred, y_test, average='weighted')}")
print(f"Recall (Kernel SVM): {recall_score(y_pred, y_test, average='weighted')}")
print(f"Runtime (Kernel SVM): {end - start}")

#TODO: use the model to make predictions

#TODO: Obtain performance metrics (runtime, accuracy, precision, recall, f1, etc.)
#for each model, make comparisons

# creating Decision Tree Classifier object with the default criteria set to gini to measure the quality of each split
# and impurity. max depth could be set to None to explore all possible values, but let's keep it at 4 for simplicity
dtc = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, random_state = 1)

# training and plotting the Decision Tree Classifier
print("-------")
print("\nDecision Tree Classifier for the Dataset (Depth = 4):")

start = timer()
dtc.fit(X_train, y_train)
end = timer()

tree.plot_tree(dtc)
plt.show()

# predictions for the decision tree classifier
y_pred = dtc.predict(X_test)

# printing the accuracy score for Decision Tree Classifier using score method
print('-------')
print('Accuracy of Decision Tree Classifier (Score Method): %.2f' % dtc.score(X_test, y_test))

# currently, there's an undefined metric warning stating that we should add 'zero_division' parameter to control the behavior
# this is unavoidable since the classifier may not differentiate between true and false positives
# setting zero_division to 1 or 0 will produce the same results (based on observation)
print("Accuracy (Decision Tree):",accuracy_score(y_pred, y_test))
print("Precision (Decision Tree):",precision_score(y_pred, y_test, average = 'weighted', zero_division = 1))
print("f1 (Decision Tree):", f1_score(y_pred, y_test, average='weighted', zero_division = 1))
print("Recall (Decision Tree):",recall_score(y_pred, y_test, average='weighted', zero_division = 1))
print(f"Runtime (Decision Tree): {end - start}")


#Implementing Random Forest Classifier
clf=RandomForestClassifier(n_estimators=500,criterion='gini')

start = timer()
clf.fit(x_train_std,np.ravel(y_train))#throwing a warning that it is expecting i d array so i used np.ravel()
end = timer()

y_pred_test=clf.predict(x_test_std)
y_pred_train = clf.predict(x_train_std)

print('-------')
#Model evaluation for testing data
print("Accuracy for training data {Random Forest Classifier}:",accuracy_score(y_train, y_pred_train))
print("Precision score for training{Random Forest classifier}:",precision_score(y_train, y_pred_train, average='micro'))
print("Recall score for training{Random Forest classifier}:",recall_score(y_train, y_pred_train, average='weighted'))
print("Precision score for training{Random Forest classifier}:",f1_score(y_train, y_pred_train, average='micro'))
print(f"Runtime (Random Forest Classifier - training): {end - start}")

print('-------')
#Model evaluation for testing data
print("Accuracy for testing data{Random Forest Classifier}:",accuracy_score(y_test, y_pred_test))
print("Precision score for testing data {Random Forest classifier}:",precision_score(y_test, y_pred_test, average='micro'))
print("Recall score for testing data{Random Forest classifier}:",recall_score(y_test, y_pred_test, average='weighted'))
print("Precision score for testing data{Random Forest classifier}:",f1_score(y_test, y_pred_test, average='micro'))
print(f"Runtime (Random Forest Classifier - testing): {end - start}")
