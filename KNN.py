from sklearn import datasets,preprocessing,neighbors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

#step1: Import the req data and check the features

iris=load_iris()
print("Iris data")
print(iris)
print("\n")

print("Iris.feature_names")
print("\n")

print("integer representing features(0=setosa,1=versicolor,2=virginica)")
print("\n")

print("3 classes of target")
print("\n")

print(iris.target_names)
print("\n")

print("total of 150 observations and 4 features")
print("\n")

print(iris.data.shape)
print("\n")

#step 2:split data and train model

X,y=iris.data[:,:],iris.target
#split (X-total dataset,y-labels, train_size:-70%-train and 30% to test)
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=0,train_size=0.7)
print("shape of train and test objects")
print("\n")
print(X_train.shape)
print(X_test.shape)
print("\n")

#shape of new y objects

print("shape of new y objects")
print("\n")
print(y_train.shape)
print(y_test.shape)
print("\n")

#trained data bfre preprocessing

print("Training data before preprocessing")
print("\n")
print(X_train)
print("\n")
print("\n")

#StandardScaler-to do scaling , fit method-for training dataset
#first 2 statements for normalisation

scaler=preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
print("display scaled data")
print("\n")
print(X_train)
print("\n")
print(X_test)

#Step 3: Train the Model and Evaluate the model .

scores = []
k_range = range(1,15)
for k in k_range:
  knn = neighbors.KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  
  
print("Prediction of Species: {}".format(y_pred))  
print("Accuracy score")
print(accuracy_score(y_test, y_pred))
print("Confusion matrix")
print(confusion_matrix(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))


