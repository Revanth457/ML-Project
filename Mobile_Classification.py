#SVM classifier
#DecisionTree classifier
#KNN classifier
#Naive's Bayes classifier
#XGB classifier
#AdaBoost classifier

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

#Read data from the train.csv file
df = pd.read_csv("train.csv")

#defining Features and Target variables
X = df.drop(columns=['price_range']) #Features
y = df['price_range'] #Target variable

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1400, test_size=600, random_state=42)





#Create a SVN classifier (for classification tasks)
svm_classifier  = SVC(kernel='linear', C=1.0, random_state=42)
#Fit the training data into the classifier 
svm_classifier.fit(X_train, y_train)

#predict the data from the SVM classifier
predictions_svm = svm_classifier.predict(X_test)
#Calculate accuracy
accuracy_svm = accuracy_score(y_test, predictions_svm)
print("Accuracy of the SVM classifier:", accuracy_svm)
#Calculate precision
precision_svm = precision_score(y_test, predictions_svm, average = 'weighted')
print("Precision of the SVM classifier:", precision_svm)
#Calculate F1 score
f1score_svm = f1_score(y_test, predictions_svm, average='weighted')
print("F1 Score of the SVM classifier:", f1score_svm)
#Calculate recall
recall_svm = recall_score(y_test, predictions_svm, average = 'weighted')
print("Recall of the SVM classifier:", recall_svm)


#Create a DecisionTree classifier (for classification tasks)
clf = DecisionTreeClassifier(random_state = 42) 
#Train the classifier on the training data
clf.fit(X_train, y_train)

#Make predictions on the test data
predictions_dt = clf.predict(X_test)
#Calculate accuracy
accuracy_dt = accuracy_score(y_test, predictions_dt)
print("Accuracy of the DecisionTree classifier:", accuracy_dt)
#Calculate precision
precision_dt = precision_score(y_test, predictions_dt, average = 'weighted')
print("Precision of the DecisionTree classifier:", precision_dt)
#Calculate F1 score
f1score_dt = f1_score(y_test, predictions_dt, average='weighted')
print("F1 Score of the DecisionTree classifier:", f1score_dt)
#Calculate recall
recall_dt = recall_score(y_test, predictions_dt, average='weighted')
print("Recall of the DecisionTree classifier:", recall_dt)


#Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
# Fit the KNN classifier to the training data
knn.fit(X_train.values, y_train.values) 

#Predict the Labels of the test data
predictions_knn = knn.predict(X_test.values)
#Calculate accuracy
accuracy_knn = accuracy_score(y_test, predictions_knn)
print("Accuracy of the KNN classifier:", accuracy_knn)
#Calculate precision
precision_knn = precision_score(y_test, predictions_knn, average = 'weighted')
print("Precision of the KNN classifier:", precision_knn)
#Calculate F1 score
f1score_knn = f1_score(y_test, predictions_knn, average = 'weighted')
print("F1 Score of the KNN classifier:", f1score_knn)
#Calculate recall
recall_knn = recall_score(y_test, predictions_knn, average='weighted')
print("Recall of the KNN classifier:", recall_knn)



#Create a Naive's Bayes classifier (for classification tasks)
clf = GaussianNB()
#Train the classifier on the training data
clf.fit(X_train, y_train)

#Make predictions on the test data 
predictions_NB  = clf.predict(X_test)
#Calculate accuracy
accuracy_nb = accuracy_score(y_test, predictions_NB)
print("Accuracy of the Navie's Bayes classifier:", accuracy_nb)
#Calculate precision
precision_nb = precision_score(y_test, predictions_NB, average='weighted')
print("Precision of the Navie's Bayes classifier:", precision_nb)
#Calculate F1 score
f1score_nb = f1_score(y_test, predictions_NB, average = 'weighted')
print("F1 Score of the Navie's Bayes classifier:", f1score_nb)
#Calculate recall
recall_nb = recall_score(y_test, predictions_NB, average  = 'weighted')
print("Recall of the Navie's Bayes classifier:", recall_nb)


#Create a XGB classifier
XGB = XGBClassifier()
# Fit the XGB classifier to the training data
XGB.fit(X_train, y_train)

# Predict the Labels of the test data
predictions_XGB = XGB.predict(X_test)
#Calculate accuracy
accuracy_xgb = accuracy_score(y_test, predictions_XGB)
print("Accuracy of the XGB classifier:", accuracy_xgb)
#Calculate precision
precision_xgb = precision_score(y_test, predictions_XGB, average='weighted') 
print("Precision of the XGB classifier:", precision_xgb)
#Calculate FI score
f1score_xgb = f1_score(y_test, predictions_XGB, average='weighted')
print("F1 Score of the XGB classifier:", f1score_xgb)
#Calculate recall
recall_xgb = recall_score(y_test, predictions_XGB, average = 'weighted')
print("Recall of the XGB classifier:", recall_xgb)



#Create a AdaBoost classifier
clf = AdaBoostClassifier()
# Fit the AdaBoost classifier to the training data
clf.fit(X_train, y_train)

# Predict the labels of the test data 
predictions_AdaB = clf.predict(X_test)
#Calculate accuracy
accuracy_ab accuracy_score(y_test, predictions_AdaB)
print("Accuracy of the AdaBoost classifier:", accuracy_ab)
#Calculate precision
precision_ab = precision_score(y_test, predictions_AdaB, average='weighted')
print("Precision of the AdaBoost classifier:", precision_ab)
#Calculate F1 score
f1score_ab = f1_score(y_test, predictions_Adaß, average weighted')
print("F1 Score of the AdaBoost classifier:", f1score_ab)
#Calculate recall
recall_ab = recall_score(y_test, predictions_Adaß, average weighted')
print("Recall of the AdaBoost classifier:", recall_ab)
