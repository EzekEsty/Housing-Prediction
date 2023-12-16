#!/usr/bin/env python
# coding: utf-8

# # IMPORTING THE DATA ON BOSTON HOUSING DATASET

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


house = pd.read_csv("C:/Users/User/Downloads/MACHINE LEARNING PRACTICE/train.csv")
house.head(10)


# In[6]:


#To drop the ID column
house.drop(["ID"], axis = "columns")


# In[8]:


house.shape


# In[12]:


sns.scatterplot(x='age', y='crim', data = house)


# In[13]:


plt.hist(house['medv'])


# In[14]:


house.isnull()


# In[15]:


house.isna()


# In[17]:


#To drop the target variable

x = house.drop(['medv'], axis = 'columns')
y = house['medv']


# In[18]:


x.head(5)


# In[19]:


y.head(5)


# In[20]:


#To split

from sklearn.model_selection import train_test_split

#To evaluate the model
from sklearn import metrics


# In[21]:


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 1234)


# # Training the Model

# In[22]:


#To import the linear regression model

from sklearn.linear_model import LinearRegression


# In[23]:


linearmodel = LinearRegression()
linearmodel.fit(x_train, y_train)


# # TO PREDICT

# In[24]:


linearmodel_pred = linearmodel.predict(x_test)


# In[25]:


linearmodel_pred


# In[26]:


#To find the Residual

y_test - linearmodel_pred


# In[27]:


MSE = metrics.mean_squared_error(y_test, linearmodel_pred)
MSE


# In[29]:


#To find the root mean squared error

RMSE = np.sqrt(MSE)
RMSE


# # TO USE THE RANDOM FOREST REGRESSOR

# In[30]:


#To train

from sklearn.ensemble import RandomForestRegressor


# In[31]:


randomforestmodel = RandomForestRegressor()


# In[32]:


randomforestmodel.fit(x_train, y_train)


# In[33]:


#To predict

randomforestmodel_pred = randomforestmodel.predict(x_test)
randomforestmodel_pred


# In[35]:


#To get the residual

y_test - randomforestmodel_pred


# In[36]:


MSE = metrics.mean_squared_error(y_test, randomforestmodel_pred)
MSE


# In[38]:


RMSE = np.sqrt(MSE)
RMSE


# In[ ]:


The Random Forest Regressor is smaller and a better model cpmpared to the Linear Regression model

