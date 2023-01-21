# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20

@author: Mujahid Shariff
"""

# step1: import the data files and libraires
import pandas as pd
df = pd.read_csv("50_Startups.csv")
df.shape
df.head()

# rename the data for your comfort
df = df.rename({'R&D Spend':'RDS','Administration':'ADMS','Marketing Spend':'MKTS'},axis=1)
df.describe()

# correlation  #finding out correlation with each variable
df.corr()

# outliers for each variable
df.boxplot("RDS", vert = False)
df.boxplot("ADMS", vert = False)
df.boxplot("MKTS", vert = False)
df.boxplot("Profit", vert = False)

# step2: split the Variables in  X and Y's

# model 1
X = df[["RDS"]] # R2: 0.947, RMSE: 9226.101

# Model 2
X = df[["RDS","ADMS"]] # R2: 0.948, RMSE: 9115.198

# Model 3
X = df[["MKTS"]] # R2: 0.559, RMSE: 26492.829

# Model 4
X = df[["MKTS","ADMS"]] # R2: 0.610, RMSE: 24927.067

# Model 5
X = df[["RDS","MKTS","ADMS"]] # R2: 0.951, RMSE: 8855.344

# Target
Y = df["Profit"]

# scatter plot between each x and Y  
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)
   
#======================================
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

# So, we will take Model 2 because in this model RMSE is low and Rsquare is high.
# our model is above than 90% it's excellent model.
