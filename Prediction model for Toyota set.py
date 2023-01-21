# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20

@author: Mujahid Shariff
"""

# step1: import the data files and libraires
import pandas as pd
df = pd.read_csv("ToyotaCorolla.csv",encoding='latin1')
df.shape
df.info()

# sort the data for your comfort
dfn = pd.concat([df.iloc[:,2:4],df.iloc[:,6:7],df.iloc[:,8:9],df.iloc[:,12:14],df.iloc[:,15:18]],axis=1)
dfn

# rename the data for your comfort
dfnw = dfn.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
dfnw

dfnw[dfnw.duplicated()]

dfnew = dfnw.drop_duplicates().reset_index(drop=True)
dfnew
dfnew.describe()

# correlation 
dfnew.corr()

# step2: split the Variables in  X and Y's

# model 1
X = dfnew[["Age"]]

# Model 2
X = dfnew[["Age","Weight"]]

# Model 3
X = dfnew[["Age","Weight","KM"]]

# Model 4
X = dfnew[["Age","Weight","KM","HP"]]

# Model 5
X = dfnew[["Age","Weight","KM","HP","QT"]]

# Model 6
X = dfnew[["Age","Weight","KM","HP","QT","Doors"]]

# Model 7
X = dfnew[["Age","Weight","KM","HP","QT","Doors","CC"]]

# Model 8
X = dfnew[["Age","Weight","KM","HP","QT","Doors","CC","Gears"]]

# Target
Y = dfnew["Price"]

# scatter plot between each x and Y  
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(dfnew)
   
#==================================
import statsmodels.api as sma
X_new = sma.add_constant(X)
lmreg = sma.OLS(Y,X_new).fit()
lmreg.summary()


# Model fitting  --> Scikit learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
mse= mean_squared_error(Y,Y_pred)
RMSE = np.sqrt(mse)
print("Root mean squarred error: ", RMSE.round(3))


# So, we will take Model 5 because in this model RMSE is low and Rsquare is high.
# our model is between 80%-90% it's Good model.