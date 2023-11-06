#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  
#pandas is used to clean and access the data


# In[2]:


data =pd.read_csv("adult.csv")


# In[3]:


data


# In[4]:


data.head()
#prints first five values


# In[5]:


data.columns


# In[6]:


data.shape
#prints(row,col)


# In[7]:


#to check for any of the data having question mark and get rid of it and replace with NAN
data.isin(['?']).sum()


# In[8]:


#replace the ? with NAN(not a number)
#one-hot encoding:

data['workclass']=data['workclass'].replace('?',np.nan)
data['occupation']=data['occupation'].replace('?',np.nan)
data['native-country']=data['native-country'].replace('?',np.nan)


# In[9]:


data.isin(['?']).sum()


# In[10]:


data.isnull().sum()
#to check missing values[empty column]


# In[11]:


data.dropna(how='any',inplace=True)
#drop all the rows that contain missing values
#data.dropna(axis,col,index,level,inplace,error)
#if inplace=False then it creates a new data frame with droped rows
#if inplace=True then it returns a original data frame with modified values


# In[12]:


#check for duplicate values of data frame
print(f"there are {data.duplicated().sum()} duplicate values")


# In[13]:


#to drop all the duplicates
data=data.drop_duplicates()


# In[14]:


print(f"there are {data.duplicated().sum()} duplicate values")


# In[15]:


data.shape


# 
# work on irrelavant features/col:

# In[16]:


data.columns


# In[17]:


data.drop(['fnlwgt','educational-num','marital-status','relationship','race'],axis=1,inplace=True)


# In[18]:


data.columns


# In[19]:


#extract X and Y from the data frame
x=data.loc[:,['age', 'workclass', 'education', 'occupation', 'gender', 'capital-gain',
       'capital-loss', 'hours-per-week', 'native-country']]
y=data.loc[:,['income']]


# In[20]:


x.head()
#Only dsc features


# In[21]:


y.head()
#target feature
#categorical data to binary values- aka one hot encoding


# In[22]:


from sklearn.preprocessing import LabelEncoder  #one way of converting categorical to binary
y=LabelEncoder().fit_transform(y)
y=pd.DataFrame(y)
y.head()
#converted to binary(the target feature i.e  income)


# In[23]:


y


# In[24]:


data.columns


# In[28]:


numeric_features=x.select_dtypes('number')
cat_features=x.select_dtypes('object') 
cat_features #prints categorical data
#now converting descriptive features into binary
#descriptive is divided two type:1)numeric 2)object data type


# In[30]:


numeric_features
#prints numerical data


# In[32]:


convert_cat_features=pd.get_dummies(cat_features) #another way of converting categorical to binary id get_dummies
convert_cat_features.shape
convert_cat_features.head()


# In[33]:


all_features=[convert_cat_features,numeric_features]
newx=pd.concat(all_features,axis=1,join='inner')
newx.shape


# In[34]:


newx.columns


# In[36]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(newx,y,test_size=0.33,random_state=42)


# In[37]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(max_depth=5)
clf.fit(x_train,y_train)


# In[38]:


#make prediction
y_pred=clf.predict(x_test)


# In[39]:


y_pred


# In[40]:


y_test.shape


# In[41]:


prediction_data=pd.DataFrame()
prediction_data['predicted_salary_class']=y_pred
prediction_data['actual_salary_class']=y_test[0].values
prediction_data


# In[42]:


prediction_data=pd.DataFrame()
prediction_data['predicted_salary_class']=y_pred
prediction_data['actual_salary_class']=y_test[0].values
prediction_data


# In[45]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))


# In[46]:


#plot the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(14,14))
plot_tree(clf,fontsize=10,filled=True)
plt.title("Decision tree based on selected features")
plt.show()


# In[ ]:




