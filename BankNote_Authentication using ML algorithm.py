

# #### Bank Note Authenticaion
#  - Data were extracted from images that were taken from genuine and forged banknote-like specimens.For digitization, an industrial camera usually used for print inspection was used.The final images have 400 x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet transform tool were used to extract features from images.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#loading dataset
df = pd.read_csv("BankNote_Authentication.csv")
print(df.head())

print(df.shape)
##checking missing values in dataset
print(df.isnull().sum())
## We can see that there is no missing values in dataset
print(df.info())

print(df['class'].value_counts())
print(df.head(10))

##Independent variable and dependent variable
X = df.iloc[:, :-1]
y = df.iloc[:,-1]

print(X.head())
print(y.head())

## Now splitting dataset into training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

##Now we importing ensemble algorithm 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

##now we train the model using fit method

classifier.fit(x_train, y_train)
##now we predict the model
y_pred = classifier.predict(x_test)

##Now we check the accuracy of the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(classification_report(y_test, y_pred))

##let's load the test file
df_test = pd.read_csv('TestFile.csv')

print(df_test.head())
#let's predict the test dataset
prediction = classifier.predict(df_test)
print(prediction)

##Now we create a Pickle file using serialization
import pickle
pickle_out = open('classifier.pkl', 'wb')

pickle.dump(classifier, pickle_out)

pickle_out.close()
##predicting model on new dataset
classifier.predict([[2,3,4,1]])








