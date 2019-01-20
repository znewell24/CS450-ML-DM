# Assignment 02 Prove
# Author: Zachary Newell
#
# Load Data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
iris = datasets.load_iris()

# Prepare Training/Test Sets
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

# skLearn nearest neighbors
neighborClassifier = KNeighborsClassifier(n_neighbors=3)
neighborClassifier.fit(X_train, y_train)
neighborPredictions = neighborClassifier.predict(X_test)
print("skLearn KNeighborsClassifier Predictions:")
print(neighborPredictions)

# implementation
class NeighborsClassifier:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
    
    def fit(self, xTrain_data, yTrain_data):
        self.xData = xTrain_data
        self.yData = yTrain_data
    
    def calcDistances(self, test):
        distances = []
        for row in self.xData:
            diff = test - row
            diff_squared = diff ** 2
            dist = sum(diff_squared)
            distances.append(dist)
        return distances
    
    def nearestNeighbors(self, test_data):
        nearest_neighbor = []
        for row in test_data:
            distances = self.calcDistances(row)
            sorted_distances = np.argsort(distances)
            nearest_neighbor.append(self.yData[sorted_distances[0]])
        return nearest_neighbor
            
    def predict(self, test_data):
        predictions = self.nearestNeighbors(test_data)
        return predictions
    
myNeighborClassifier = NeighborsClassifier(n_neighbors=3)
myNeighborClassifier.fit(X_train, y_train)
myNeighborPredictions = myNeighborClassifier.predict(X_test)
print("My Predictions:")
print(myNeighborPredictions)
    
 
    