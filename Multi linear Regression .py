#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


# In[43]:


dataset = pd.read_csv("50_Startups.csv")
X= dataset.iloc[:,:-1]
y = dataset.iloc[:,4]


# In[44]:


dataset.head()


# In[45]:


y.head()


# In[8]:


y.head()


# In[46]:


state = pd.get_dummies(X['State'],drop_first=True)


# In[47]:


state.head()


# In[48]:


X=X.drop('State',axis=1)


# In[49]:


X.head()


# In[50]:


X=pd.concat([X,state],axis=1)


# In[36]:


y.head()


# In[51]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[52]:


y_train.head()


# In[53]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[58]:


y_pred=regressor.predict(X_test)


# In[59]:


y_pred


# In[60]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)


# In[61]:


score


# In[ ]:




