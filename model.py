import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

df=pd.read_csv('iris.csv')
X=df.iloc[:,:4]
y=df.iloc[:,4]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn import tree

classifier=tree.DecisionTreeClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

pickle.dump(classifier,open('model.pkl','wb'))
