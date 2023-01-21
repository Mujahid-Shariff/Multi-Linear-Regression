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

#R-square for every model
#R squared values
(lmreg.rsquared.round(3),lmreg.rsquared_adj.round(3)) 

#model 1, R square values - 0.947
#model 2, R square values - 0.948
#model 3, R square values - 0.559
#model 4, R square values - 0.610
#model 5, R square values - 0.951


# So, we will take Model 2 because in this model RMSE is low and Rsquare is high.
# our model is above than 90% it's excellent model.

#Residual Analysis
## Test for Normality of Residuals (Q-Q Plot)
import matplotlib.pyplot as plt
import statsmodels.api as sm
qqplot=sm.qqplot(lmreg.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


lmreg.resid.hist()
lmreg.resid

list(np.where(lmreg.resid>10))

## Residual Plot for Homoscedasticity

lmreg.fittedvalues
lmreg.resid

# finding out patterns using plot
import matplotlib.pyplot as plt
plt.scatter(lmreg.fittedvalues,lmreg.resid)
plt.title('Residual Plot')
plt.xlabel('Fitted values')
plt.ylabel('residual values')
plt.show()

## Cook’s Distance

model_influence = lmreg.get_influence()
(cooks, pvalue) = model_influence.cooks_distance

cooks = pd.DataFrame(cooks)
cooks[0].describe()

#obtain Cook's distance for each observation
cooks = lmreg.cooks_distance

#display Cook's distances
print(cooks)

#Visualize Cook’s Distances
import matplotlib.pyplot as plt
plt.scatter(X, cooks[0])
plt.xlabel('X')
plt.ylabel('Cooks Distance')
plt.show()

#Plot the influencers values using stem plot
import matplotlib.pyplot as plt
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df)), np.round(cooks[0], 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()

## High Influence points
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(lmreg)
plt.show()

k = df.shape[1]
n = df.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff

# points which are greater than leverage value treated as influencer observations
cooks[0][cooks[0]>leverage_cutoff]
