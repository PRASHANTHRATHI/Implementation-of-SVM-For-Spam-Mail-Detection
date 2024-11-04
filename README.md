# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements.
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Print all the outputs.
6. End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Prashanth.K
RegisterNumber:  212223230152
*/
```

```
import pandas as pd
data= pd.read_csv("C:/Users/admin/Desktop/INTR MACH/spam.csv", encoding= 'Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test , y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test= cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train , y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy= metrics.accuracy_score(y_test, y_pred)
accuracy

from sklearn.metrics import confusion_matrix, classification_report
con= confusion_matrix(y_test, y_pred)
print(con)cl=classification_report(y_test,y_pred)
print(cl)

```

## Output:
# data_head():

![image](https://github.com/user-attachments/assets/fc6d76bd-1eda-4a94-8979-53b0f9c3808f)

# data.isnull().sum():

![image](https://github.com/user-attachments/assets/fe634ce5-5f41-4715-b407-f989bdf0c494)

# accuracy:

![image](https://github.com/user-attachments/assets/52c0b3ba-7b08-4d15-851f-d0e293b38f00)

# Confusion matrix:

![image](https://github.com/user-attachments/assets/410e454e-d070-47a9-9b31-3f3ef535324e)

# Classification report:

![image](https://github.com/user-attachments/assets/527383e3-35e1-43ea-b312-f6768f62f0d4)








## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
