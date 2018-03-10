from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Data and labels
#[Sepal.Length, Sepal.Width, Petal.Length,Petal.Width]
X = [[5.1, 3.5, 1.4, 0.2], [4.9,3,1.4,0.2], [4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],[5,3.6,1.4,2],
	[7,3.2,4.7,1.4],[6.4,3.2,4.5,1.5], [6.9,3.1,4.9,1.5], [5.5,2.3,4,1.3],[6.5,2.8,4.6,1.5],
	[6.3,3.3, 6,2.5],[5.8,2.7, 5.1,1.9],[7.1,3,5.9,2.1], [6.3,2.9, 5.6,1.8],[6.5,3,5.8,2.2]]


Y = ['setosa', 'setosa', 'setosa', 'setosa', 'setosa',
	 'versicolor', 'versicolor', 'versicolorle', 'versicolor', 'versicolorle',
	 'virginica', 'virginica','virginica','virginica','virginica']

# Classifiers
# using the default values for all the hyperparameters
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()

# Training the models
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_KNN.fit(X, Y)

# Testing using the same data
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

pred_per = clf_perceptron.predict(X)
acc_per = accuracy_score(Y, pred_per) * 100
print('Accuracy for perceptron: {}'.format(acc_per))

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

# The best classifier from svm, per, KNN
index = np.argmax([acc_svm, acc_per, acc_KNN])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN'}
print('Best species classifier is {}'.format(classifiers[index]))
