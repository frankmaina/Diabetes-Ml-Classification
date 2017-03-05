# import packages
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# import training data set
df = pd.read_csv("train/data.csv")

# treat all 0 values that are not in the pregnanceies column = as outliers
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = df[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace('0', -99999)

# impoty features
X = np.array(df.drop(['Class'], 1))

# import labels
y = np.array(df['Class'])

# split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# initiate the classifier and train
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

# test the level of accuracy using 10% of out training data
accuracy = clf.score(X_test, y_test)

print("The Classifier accuracy level is at %s" % (accuracy * 100))
