folder="C:\\Users\\Admin\\OneDrive\\Documents\\VSCode Practice\\Practice\\ML\\"
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
diabetes=pd.read_csv(folder+'diabetes.csv')
diabetes.head() #shows only the first 5 lines
# print(diabetes.head())
diabetes.columns
zero_unaccepted=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

for column in zero_unaccepted:
    diabetes[column] = diabetes[column].replace(0, np.NaN)
    mean = int(diabetes[column].mean(skipna=True))
    diabetes[column] = diabetes[column].fillna(mean)


x=diabetes.iloc[:,0:8]    #columns from 0 to 7
y=diabetes.iloc[:,8]   #only the 8th column
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=0, test_size=0.2)   #part of sklearn   testsize=0.2 i.e.20% of data set aside to test

# Convert to DataFrame
X_train = pd.DataFrame(X_train, columns=x.columns)
X_test = pd.DataFrame(X_test, columns=x.columns)

#-----Feature Scaling-----
sc_X=StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier=KNeighborsClassifier(n_neighbors=11, p=2,metric="euclidean")
classifier.fit(X_train, Y_train)

y_pred=classifier.predict(X_test)
# print(y_pred)

cm=confusion_matrix(Y_test,y_pred)
print(cm)
print(f1_score(Y_test,y_pred))
print(accuracy_score(Y_test,y_pred))





