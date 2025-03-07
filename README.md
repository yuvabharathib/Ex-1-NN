<H3>ENTER YOUR NAME: Yuvabharathi.B</H3>
<H3>ENTER YOUR REGISTER NO: 212222230181</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
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
```python
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```
```python
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
```
```python
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
```
```python
df.isnull().sum()
```
```python
df.duplicated()
```
```python
df.describe()
```
```python
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
```
```python
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
```
```python
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```
## OUTPUT:
#### DATASET:
![Screenshot 2025-03-07 114502](https://github.com/user-attachments/assets/6b8d107b-ae72-4e35-8c37-e9af132fbe15)

#### DROPPING THE UNWANTED DATASET:
![Screenshot 2025-03-07 114538](https://github.com/user-attachments/assets/6737abb1-79fa-40cd-b762-b104aa0d2bff)

#### CHECKING NULL VALUES:

![Screenshot 2025-03-07 114603](https://github.com/user-attachments/assets/a7b40b11-7098-4782-b228-7d5a12f51293)

#### CHECKING FOR DUPLICATION:
![Screenshot 2025-03-07 114741](https://github.com/user-attachments/assets/905b6729-bd4c-4d7b-9a69-c96aef80636d)

#### DESCRIBING THE DATASET:
![Screenshot 2025-03-07 114640](https://github.com/user-attachments/assets/443d104d-7003-4256-a1b7-5cb72ad9ee98)

#### SCALING THE DATASET:
![Screenshot 2025-03-07 115509](https://github.com/user-attachments/assets/92b9c751-4c74-4d1f-8ddc-d4e6de991422)

#### X FEATURES:
![Screenshot 2025-03-07 115313](https://github.com/user-attachments/assets/22c5c2dd-351b-4446-84d4-022ec2e15a7e)

#### Y FEATURES:
![Screenshot 2025-03-07 114314](https://github.com/user-attachments/assets/7ca7e06d-b681-42ad-84ff-a630f11e8310)

#### SPLITTING THE TRAINING AND TESTING DATASET:
![Screenshot 2025-03-07 114151](https://github.com/user-attachments/assets/b58409ec-2073-4323-a784-2f188a7f62f0)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


