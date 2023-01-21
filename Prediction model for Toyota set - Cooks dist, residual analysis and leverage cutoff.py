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

#R-square for every model
#R squared values
(lmreg.rsquared.round(3),lmreg.rsquared_adj.round(3)) 

#model 1, R square values - 0.768
#model 2, R square values - 0.804
#model 3, R square values - 0.847
#model 4, R square values - 0.861
#model 5, R square values - 0.861
#model 6, R square values - 0.861
#model 7, R square values - 0.862
#model 8, R square values - 0.863

# So, we will take Model 5 because in this model RMSE is low and Rsquare is high.
# our model is between 80%-90% it's Good model.

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
