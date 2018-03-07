#import scikit and specify the module we want to import
from sklearn import tree

#clf as Decision Tree 
clf = tree.DecisionTreeClassifier()

#X is a list of lists (a text string)
#[Sepal.Length, Sepal.Width, Petal.Length,Petal.Width]
X = [[5.1, 3.5, 1.4, 0.2], [4.9,3,1.4,0.2], [4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],[5,3.6,1.4,2],
	[7,3.2,4.7,1.4],[6.4,3.2,4.5,1.5], [6.9,3.1,4.9,1.5], [5.5,2.3,4,1.3],[6.5,2.8,4.6,1.5],
	[6.3,3.3, 6,2.5],[5.8,2.7, 5.1,1.9],[7.1,3,5.9,2.1], [6.3,2.9, 5.6,1.8],[6.5,3,5.8,2.2]]

#labeled data 
Y = ['setosa', 'setosa', 'setosa', 'setosa', 'setosa',
	 'versicolor', 'versicolor', 'versicolorle', 'versicolor', 'versicolorle',
	 'virginica', 'virginica','virginica','virginica','virginica']


#fit/trains the model on our sample data 
clf = clf.fit(X, Y)

#store our prediction to determine whether the parameters entered are either male or female. 
prediction = clf.predict([[6, 4, 4, 1.5]])

#print our prediction
print(prediction)
