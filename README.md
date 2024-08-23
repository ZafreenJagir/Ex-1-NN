<H3>ENTER YOUR NAME : ZAFREEN J</H3> 
<H3>ENTER YOUR REGISTER NO : 212223040252</H3>
<H3>EX. NO.1</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```

![Screenshot 2024-08-23 210606](https://github.com/user-attachments/assets/d696168d-0f0b-4514-896a-3233266a5c5e)


```
df.isnull().sum()
```

![Screenshot 2024-08-23 210712](https://github.com/user-attachments/assets/2f580f64-1819-4d16-aa62-49d4b74acd21)


```
df.duplicated()
```

![Screenshot 2024-08-23 211144](https://github.com/user-attachments/assets/c7b169d7-7574-4b26-aaf9-e9812f73a429)



```
print(df['CreditScore'].describe())
```

![Screenshot 2024-08-23 211254](https://github.com/user-attachments/assets/233dd336-df2b-4303-a916-2837e5cf05a4)



```
df.info()
```

![Screenshot 2024-08-23 211526](https://github.com/user-attachments/assets/9334db11-9036-44e5-ba7b-d73f22d45359)


```
df.drop(['Surname','CustomerId','Geography','Gender'],axis=1,inplace=True)
df
```

![Screenshot 2024-08-23 211731](https://github.com/user-attachments/assets/6afea769-48f1-4c61-afdb-5d230885ecde)


```
scaler=MinMaxScaler()
df=pd.DataFrame(scaler.fit_transform(df))
df
```


![Screenshot 2024-08-23 211824](https://github.com/user-attachments/assets/940fb085-ddde-4d44-9de4-93d710c7bc7d)


```
X = df.iloc[:, :-1].values
print(X)
```


![Screenshot 2024-08-23 211959](https://github.com/user-attachments/assets/f63c7d37-7d10-4a7e-aafe-689ed7c0df85)


```
y = df.iloc[:,-1].values
print(y)
```

![Screenshot 2024-08-23 212058](https://github.com/user-attachments/assets/45baeb88-a113-4d38-b17d-f0cd4de23d5a)


```
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)
```

![Screenshot 2024-08-23 212209](https://github.com/user-attachments/assets/1a8da19f-1632-45ad-907f-0ff61098dba6)


```
print(X_train)
print(len(X_train))
```

![Screenshot 2024-08-23 212258](https://github.com/user-attachments/assets/0958c415-e806-48e9-8bdf-311d4aba2c97)


```
print(X_test)
print(len(X_test))
```

![Screenshot 2024-08-23 212603](https://github.com/user-attachments/assets/5e69c228-4503-4949-a955-7972a88057d7)






## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


