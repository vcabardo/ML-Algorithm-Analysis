import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from timeit import default_timer as timer
from pylab import rcParams
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#these lists will store the accuracies and runtimes of all tested classifiers,
#to be visualized in a matplotlib graph
accuracies = []
runtimes = []

df = pd.read_csv("Dry_Bean_Dataset.csv")

#separate the features and the class labels
X=df[df.columns[0:16]]
y=df[['Class']]

#separate the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

#perform feature scaling
scaler = StandardScaler()
x_train_std = scaler.fit_transform(X_train)
x_test_std = scaler.fit_transform(X_test)

#classifier 1: Linear SVM
svm = SVC(kernel='linear', C=0.01, random_state=1)

start = timer()
svm.fit(x_train_std, np.ravel(y_train))
end = timer()

y_pred = svm.predict(x_test_std)

#Overall performance metrics of Linear SVM - commented out for cleaner output
# print("-------")
# print(f"Accuracy (Linear SVM): {accuracy_score(y_pred, y_test)}")
# print(f"Precision (Linear  SVM): {precision_score(y_pred, y_test, average='weighted')}")
# print(f"f1 (Linear SVM): {f1_score(y_pred, y_test, average='weighted')}")
# print(f"Recall (Linear SVM): {recall_score(y_pred, y_test, average='weighted')}")
# print(f"Runtime (Linear SVM): {end - start}")

#obtain the classification report to obtain precision, recall, and f1 metrics
#on a per class basis
report = classification_report(y_test, y_pred, output_dict=True, target_names=pd.unique(np.ravel(df[['Class']])))

#record the precision, recall, and f1 score for each class,
#this data will be visualized in a matplotlib graph
precisions_linearsvm = []
recalls_linearsvm  = []
f1s_linearsvm  = []
classes = ["SEKER", "BARBUNYA", "BOMBAY", "CALI", "HOROZ", "SIRA", "DERMASON", "Macro Average", "Weighted Average"]

for k in report:
  if k != 'accuracy':
    precisions_linearsvm.append(report[k]['precision'])
    recalls_linearsvm.append(report[k]['recall'])
    f1s_linearsvm.append(report[k]['f1-score'])

#accuracy and runtime cannot be measured on a per-class basis -
#we append them separately here
accuracies.append(accuracy_score(y_pred, y_test))
runtimes.append(end - start)

#classifier 2: Kernel SVM
svm = SVC(kernel='rbf', C=0.01, random_state=1)

start = timer()
svm.fit(x_train_std, np.ravel(y_train))
end = timer()

y_pred = svm.predict(x_test_std)

#Overall performance metrics of Kernel SVM - commented out for cleaner output
# print("-------")
# print(f"Accuracy (Kernel SVM): {accuracy_score(y_pred, y_test)}")
# print(f"Precision (Kernel SVM): {precision_score(y_pred, y_test, average='weighted')}")
# print(f"f1 (Kernel SVM): {f1_score(y_pred, y_test, average='weighted')}")
# print(f"Recall (Kernel SVM): {recall_score(y_pred, y_test, average='weighted')}")
# print(f"Runtime (Kernel SVM): {end - start}")

#obtain the classification report to obtain precision, recall, and f1 metrics
#on a per class basis
report = classification_report(y_test, y_pred, output_dict=True, target_names=pd.unique(np.ravel(df[['Class']])))

#record the precision, recall, and f1 score for each class,
#this data will be visualized in a matplotlib graph
precisions_kernelsvm = []
recalls_kernelsvm = []
f1s_kernelsvm = []

for k in report:
  if k != 'accuracy':
    precisions_kernelsvm.append(report[k]['precision'])
    recalls_kernelsvm.append(report[k]['recall'])
    f1s_kernelsvm.append(report[k]['f1-score'])

#accuracy and runtime cannot be measured on a per-class basis -
#we append them separately here
accuracies.append(accuracy_score(y_pred, y_test))
runtimes.append(end - start)


#classifier 3: Decision Tree
dtc = DecisionTreeClassifier(criterion = 'gini', max_depth = 8, random_state = 1)

start = timer()
dtc.fit(X_train, y_train)
end = timer()

y_pred = dtc.predict(X_test)

# Overall performance metrics of Kernel SVM - commented out for cleaner output
# print("Accuracy (Decision Tree):",accuracy_score(y_pred, y_test))
# print("Precision (Decision Tree):",precision_score(y_pred, y_test, average = 'weighted', zero_division = 1))
# print("f1 (Decision Tree):", f1_score(y_pred, y_test, average='weighted', zero_division = 1))
# print("Recall (Decision Tree):",recall_score(y_pred, y_test, average='weighted', zero_division = 1))
# print(f"Runtime (Decision Tree): {end - start}")

#obtain the classification report to obtain precision, recall, and f1 metrics
#on a per class basis
report = classification_report(y_test, y_pred, output_dict=True, target_names=pd.unique(np.ravel(df[['Class']])))

#record the precision, recall, and f1 score for each class,
#this data will be visualized in a matplotlib graph
precisions_decision = []
recalls_decision  = []
f1s_decision = []
classes = ["SEKER", "BARBUNYA", "BOMBAY", "CALI", "HOROZ", "SIRA", "DERMASON", "Macro Average", "Weighted Average"]

for k in report:
  if k != 'accuracy':
    precisions_decision.append(report[k]['precision'])
    recalls_decision.append(report[k]['recall'])
    f1s_decision.append(report[k]['f1-score'])

#accuracy and runtime cannot be measured on a per-class basis -
#we append them separately here
accuracies.append(accuracy_score(y_pred, y_pred))
runtimes.append(end - start)


#classifier 4: Decision Tree
clf=RandomForestClassifier(n_estimators=500,criterion='gini')

start = timer()
clf.fit(x_train_std,np.ravel(y_train))
end = timer()

y_pred_test=clf.predict(x_test_std)
y_pred_train = clf.predict(x_train_std)

# Overall performance metrics of RandomForestClassifier - commented out for cleaner output
# print('-------')
# #Model evaluation for testing data
# print("Accuracy for testing data{Random Forest Classifier}:",accuracy_score(y_test, y_pred_test))
# print("Precision score for testing data {Random Forest classifier}:",precision_score(y_test, y_pred_test, average='micro'))
# print("Recall score for testing data{Random Forest classifier}:",recall_score(y_test, y_pred_test, average='weighted'))
# print("Precision score for testing data{Random Forest classifier}:",f1_score(y_test, y_pred_test, average='micro'))
# print(f"Runtime (Random Forest Classifier - testing): {end - start}")

#obtain the classification report to obtain precision, recall, and f1 metrics
#on a per class basis
report = classification_report(y_test, y_pred_test, output_dict=True, target_names=pd.unique(np.ravel(df[['Class']])))

#record the precision, recall, and f1 score for each class,
#this data will be visualized in a matplotlib graph
precisions_rfc = []
recalls_rfc = []
f1s_rfc = []

for k in report:
  if k != 'accuracy':
    precisions_rfc.append(report[k]['precision'])
    recalls_rfc.append(report[k]['recall'])
    f1s_rfc.append(report[k]['f1-score'])

#accuracy and runtime cannot be measured on a per-class basis -
#we append them separately here
accuracies.append(accuracy_score(y_pred, y_pred_test))
runtimes.append(end - start)



#Plot the precision scores for each class
#https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
X = np.arange(9)
plt.bar(X - 0.30, precisions_linearsvm, color = 'r', align='edge', width = 0.15)
plt.bar(X - 0.15, precisions_kernelsvm, color = 'b', align='edge',width = 0.15)
plt.bar(X + 0.00, precisions_rfc, color = 'g', align='edge',width = 0.15)
plt.bar(X + 0.15, precisions_decision, color = 'y',align='edge', width = 0.15)
plt.xticks(X, classes)
plt.title('Report: Precision Scores')
plt.legend(labels=['Linear SVM', 'Kernel SVM', 'RandomForestClassifier', 'Decision Tree'])
plt.ylabel('Classification Rate (%)')
plt.xlabel('Class')
rcParams['figure.figsize'] = 20, 20
plt.show()

#Plot the recall scores for each class
#https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
X = np.arange(9)
plt.bar(X - 0.30, recalls_linearsvm, color = 'r', align='edge', width = 0.15)
plt.bar(X - 0.15, recalls_kernelsvm, color = 'b', align='edge',width = 0.15)
plt.bar(X + 0.00, recalls_rfc, color = 'g', align='edge',width = 0.15)
plt.bar(X + 0.15, recalls_decision, color = 'y',align='edge', width = 0.15)
plt.xticks(X, classes)
plt.title('Report: Recall Scores')
plt.legend(labels=['Linear SVM', 'Kernel SVM', 'RandomForestClassifier', 'Decision Tree'])
plt.ylabel('Classification Rate (%)')
plt.xlabel('Class')
rcParams['figure.figsize'] = 20, 20
plt.show()

#Plot the f1 scores for each class
#https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
X = np.arange(9)
plt.bar(X - 0.30, f1s_linearsvm, color = 'r', align='edge', width = 0.15)
plt.bar(X - 0.15, f1s_kernelsvm, color = 'b', align='edge',width = 0.15)
plt.bar(X + 0.00, f1s_rfc, color = 'g', align='edge',width = 0.15)
plt.bar(X + 0.15, f1s_decision, color = 'y',align='edge', width = 0.15)
plt.xticks(X, classes)
plt.title('Report: F1 Scores')
plt.legend(labels=['Linear SVM', 'Kernel SVM', 'RandomForestClassifier', 'Decision Tree'])
plt.ylabel('Classification Rate (%)')
plt.xlabel('Class')
rcParams['figure.figsize'] = 20, 20
plt.show()

#Plot the accuracy scores for each classifier
#https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
X = np.arange(4)
plt.bar(X, accuracies, width = 0.25)
plt.xticks(X, ['Linear SVM', 'Kernel SVM', 'Decision Tree', 'RandomForestClassifier'])
plt.ylabel('Accuracy')
plt.xlabel('Classifier Type')
rcParams['figure.figsize'] = 20, 20
plt.show()

#Plot the runtime for each classifier
#https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
X = np.arange(4)
plt.bar(X, runtimes, width = 0.25)
plt.xticks(X, ['Linear SVM', 'Kernel SVM', 'Decision Tree', 'RandomForestClassifier'])
plt.ylabel('Runtime')
plt.xlabel('Classifier Type')
rcParams['figure.figsize'] = 20, 20
plt.show()
