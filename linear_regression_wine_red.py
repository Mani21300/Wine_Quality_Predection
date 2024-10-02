

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.__version__

#!pip show np

# pip install - r requirements.txt

"""$Step-2$

**Read the data**
"""

# read the dataset

df = pd.read_csv("winequality_red.csv")
df.head()

"""**Objective**

- Based on years of experience , estimate the salary of an employee
"""

df.shape

df.columns

df.isnull().sum()

df.dtypes

"""- We divide data into two parts i.e input data and output data

- input data = X; output data=y

- Again we divide input data into two parts i.e train and test

- input train data= x_train; input test data= x_test

- similarly we divide output data into two parts i.e train and test

- output train data= y_train; output test data= y_test

- Model development happens on train data i.e x_train and y_train

- Model will predict by passing x_test data, these are called y_predictions

- y_predictions will compare with y_test , this is called test accuracy/ test error
"""

#x_train   y_train
#1           1
#2           4
#3           9
#4           16

#x_test    y_test
#5         25

#develope a model (1,1) (2,4) (3,9) (4,16)
#model will predict by passing 5 , y_predictions  we need to compare with y_test

"""$Step-3$

**divide data into input and output data**
"""

df.columns

X=df.drop('quality',axis=1)
y=df['quality']

"""$Step-4$

**Divide data into train and test**
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,  # Input data
                                                  y,  # output data
                                                  random_state=1234, # it select random samples
                                                  test_size=0.30)

X_train.shape, X_test.shape

y_train.shape, y_test.shape

X_train

"""$Step-5$

**Model development**
"""

# Model development happens using train data
# X_train    y_train
#from sklearn.linear_model import LinearRegression
#LR=LinearRegression()

X_train.shape

(21,2)

df.shape   # 30 rows   2 columns

df.ndim # number of dimensions

X_train.ndim
# 1 dimension means 1 column only
# 2 dimension means 2 column only
# when you have only 1 coulmn, the shape will not show the coulumn
# (21,) it is only one column data having 21 observations
# (9,) it is one column data having 9 observation
# (30,2) it is 2 column data having 30 observation
# Reshape the data if you have only one column

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,
       y_train)

"""$Step-6$

**Model predictions**
"""

# Model predictions happens X_test
y_predictions=LR.predict(X_test)

y_predictions

"""$Step-7$

**Model evaluation**
"""

# RMSE
# MSE
# MAE
# R-square

from sklearn.metrics import r2_score,mean_squared_error

R2=r2_score(y_test,y_predictions)
MSE=mean_squared_error(y_test,y_predictions)
RMSE=np.sqrt(MSE)
#accuracy_score(y_test,y_predictions) # it is a regression tech
print("R-sqaure:",R2)
print("MSE:",MSE)
print("RMSE:",RMSE)

# Suppose your original salary is 50k
# Our model will expecting either 44k  or  56k

"""$Step-8$

**Finding coeffiecnt and Intercept**

- Coefficient means b0 ,b1 ,b2....

- Coeffiecints depends on number of input features

- In this data we have only one column as input i.e. Years of Experience

- So we will get only one coeffiecnt
"""

LR.coef_
print("The coeffiecnt of Years_of_experience is:",LR.coef_)

LR.intercept_

#Regression_equation=LR.intercept_+LR.coef_*'YearsExperience'
#Regression_equation

"""$Step-9$

**Plot the regression line**

- In order to plot regression line

- We need to undertsand the two plots

- Orginal data plot i.e input data(X) vs output data(y)

- Regression plot i.e input data (X) vs predictions of regression model by passing input data (X)
"""

# Draw the regression line on original data vs predictions on original data

#original_y_predictions=LR.predict(X.array.reshape(-1,1))
#plt.scatter(X,y,label='original data')  # Original plot
#plt.plot(X,original_y_predictions,color='red') # Regression plot

"""$Step-10$

**Stas.OLS method**
"""

from statsmodels.api import OLS
OLS(y_train,X_train).fit().summary()



## All together

################################## Data into two parts############################################
X=df['YearsExperience']
y=df['Salary']


################################ Train test split #################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,  # Input data
                                                  y,  # output data
                                                  random_state=1234, # it select random samples
                                                  test_size=0.30)

#########################Model predictions happens X_test############################################
y_predictions=LR.predict(X_test.array.reshape(-1, 1))


######################### Metrics######################################################################

from sklearn.metrics import r2_score,mean_squared_error
R2=r2_score(y_test,y_predictions)
MSE=mean_squared_error(y_test,y_predictions)
RMSE=np.sqrt(MSE)
#accuracy_score(y_test,y_predictions) # it is a regression tech
print("R-sqaure:",R2)
print("MSE:",MSE)
print("RMSE:",RMSE)

"""$Step-11$:
    
**Save the model**
"""

import pickle
pickle.dump(LR,
            open('linear_wine_model.pkl','wb'))

#Model name=LR
#In which name the model is saving: linear_slaary_model
# extenstion: Pickle
# wb: write in bytes

"""$Step-12$:

**Load the model**
"""

# Loading model to compare the results
model = pickle.load(open('linear_wine_model.pkl','rb'))
model

"""$Step-13$:
    
**Predictions**
"""

X_test

X_test.values

len(X_test.columns)

model.predict([[1,2,3,4,5,6,7,8,9,10,11]])
# the input columns are 11
# so we need pass 11 values as list

model.predict(X_test)

import os
os.getcwd()

