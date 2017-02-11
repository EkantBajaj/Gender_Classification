# import dependencies
import pandas as pd
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Read the data

data = pd.read_csv('adults.txt',sep = ',')

# Change the string labels to the numeric labels

for labels in ['education','occupation','relationship','race']:
	data[labels] = LabelEncoder().fit_transform(data[labels])

# Setting up features and labels

X = data[['relationship','occupation','race']]
y = data['sex'].values.tolist()

# splitting data into test and train set

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .3)

# creating classification model

clf = tree.DecisionTreeClassifier()

# train the model on tain data set

clf = clf.fit(X_train,y_train)

# doing prediction

prediction = clf.predict(X_test)

# finding accuracy

accuracy = clf.score(X_test,y_test)
print('accuracy is', accuracy)

# making confusion matrix
cm = confusion_matrix(prediction,y_test)
print(cm)