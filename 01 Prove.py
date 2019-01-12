# Assignment 01 Prove
# Author: Zachary Newell
#
# 1. Load Data
from sklearn.model_selection import train_test_split
from sklearn import datasets
iris = datasets.load_iris()

# 2. Prepare Training/Test Sets
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

print("Shape of training arrays:")
print(X_train.shape)
print(y_train.shape)

# 3. Use an Exisiting Algorithm to Create a Model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# 4. Use That Model to Make Perdictions
targets_predicted = classifier.predict(X_test)
print("Prediction:")
print(targets_predicted)


# 5. Implement Your Own New "Algorithm"
class HardCodedClassifier:
    def fit(self, xDataset, yDataset):
        pass
    
    def predict(self, test_data):
        n = len(test_data)
        list_of_zeroes = [0] * n
        return list_of_zeroes
        
hardCodedClassifer = HardCodedClassifier()
hardCodedClassifer.fit(X_train, y_train)

hardCodedPrediction = hardCodedClassifer.predict(X_test)
print("Hard Coded Classifier prediction:")
print(hardCodedPrediction)
    