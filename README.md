<H3>ENTER YOUR NAME : RAJA GOPAL V</H3>
<H3>ENTER YOUR REGISTER NO : 212223240134</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 22/08/2025</H3>
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
```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))

```


## OUTPUT:
### Dataset:
<img width="1329" height="474" alt="image" src="https://github.com/user-attachments/assets/26b3626b-667e-49d3-8875-fee6425de312" />

### X Values:
<img width="663" height="142" alt="image" src="https://github.com/user-attachments/assets/ba9288ee-63aa-4eb4-87b6-b5762b37da08" />

### Y Values:
<img width="324" height="31" alt="image" src="https://github.com/user-attachments/assets/14236794-9f96-4317-9cae-b36778021c1c" />

### Null Values:
<img width="297" height="513" alt="image" src="https://github.com/user-attachments/assets/d3acada7-aaab-46e9-92fc-c3e8f1253270" />

### Duplicated Values:
<img width="294" height="464" alt="image" src="https://github.com/user-attachments/assets/ac3fa7d9-8883-482a-9781-a3932cb9edfa" />

### Description:
<img width="1364" height="298" alt="image" src="https://github.com/user-attachments/assets/6c883021-d154-4ed9-9994-1d179d5ebb99" />

### Normalized Dataset:
<img width="762" height="482" alt="image" src="https://github.com/user-attachments/assets/6a81e601-8152-4469-8f72-6436d0bc465f" />

### Training Data:
<img width="765" height="225" alt="image" src="https://github.com/user-attachments/assets/ccf0756e-f3df-49f4-a80c-f297d0837d45" />

### Testing Data:
<img width="730" height="228" alt="image" src="https://github.com/user-attachments/assets/ee1b0bad-a8a8-4dd8-9f5d-e8ab8ad32331" />
<img width="339" height="47" alt="image" src="https://github.com/user-attachments/assets/79cc620a-c367-4a9e-8508-6412bbb35cd4" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


