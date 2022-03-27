# TODO: import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold

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
svm.fit(x_train_std, np.ravel(y_train))

y_pred = svm.predict(x_test_std)

print("-------")
print(f"Accuracy (Kernel SVM): {accuracy_score(y_pred, y_test)}")
print(f"Precision (Kernel SVM): {precision_score(y_pred, y_test, average='weighted')}")
print(f"f1 (Kernel SVM): {f1_score(y_pred, y_test, average='weighted')}")
print(f"Recall (Kernel SVM): {recall_score(y_pred, y_test, average='weighted')}")

svm = SVC(kernel='rbf', C=0.01, random_state=1)
svm.fit(x_train_std, np.ravel(y_train))

y_pred = svm.predict(x_test_std)

print("-------")
print(f"Accuracy (Kernel SVM): {accuracy_score(y_pred, y_test)}")
print(f"Precision (Kernel SVM): {precision_score(y_pred, y_test, average='weighted')}")
print(f"f1 (Kernel SVM): {f1_score(y_pred, y_test, average='weighted')}")
print(f"Recall (Kernel SVM): {recall_score(y_pred, y_test, average='weighted')}")

#TODO: use the model to make predictions

#TODO: Obtain performance metrics (runtime, accuracy, precision, recall, f1, etc.)
#for each model, make comparisons
