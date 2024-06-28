#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# In[24]:


df_test = pd.read_csv('C:/Users/siddh/Desktop/Genre Classification Dataset/test_data.txt', sep=":::", header = 0 , engine = 'python')
df_train = pd.read_csv('C:/Users/siddh/Desktop/Genre Classification Dataset/train_data.txt',sep=":::", header = 0 , engine = 'python')
df_train.columns = ['Serial Number', 'movie_name', 'Category', 'confession']
df_test.columns = ['Serial Number', 'movie_name' , 'confession']


# In[10]:


df_test.head()


# In[11]:


df_train.head()


# In[12]:


df_test.info()


# In[13]:


df_train.info()


# In[14]:


df_train.describe()


# In[15]:


df_test.describe()


# In[16]:


df_test.isnull().sum()


# In[17]:


df_train.isnull().sum()


# In[18]:


df_train.count()


# In[19]:


df_test.count()


# In[20]:


df_train.iloc[0:3]


# In[21]:


df_test.shape


# In[22]:


df_train.shape


# In[63]:


plt.figure(figsize=(14,10))
sns.countplot(x='Category', data=df_train)
plt.xlabel('Movie Category')
plt.ylabel('Count')
plt.title('Movie Genre')
plt.xticks(rotation=90);
plt.show()


# In[65]:


# sns.displot(df_train.Category, kde =True, color = "black")
plt.xticks(rotation=80);


# In[31]:


sns.displot(df_train.Category, kde=False, color = "grey")
plt.xticks(rotation=80);


# In[32]:


plt.figure(figsize = (14,10))
count1 = df_train.Category.value_counts()
sns.barplot(x = count1, y = count1.index, orient = 'h')
plt.xlabel('Count')
plt.ylabel('Categories')
plt.title('Movie Genre')
plt.xticks(rotation=80)
plt.show()


# In[33]:


plt.figure(figsize = (14,10))
count1 = df_train.Category.value_counts()
sns.barplot(x = count1, y = count1.index, orient = 'h')
plt.xlabel('Count', fontsize = 18, fontweight = 'bold')
plt.ylabel('Categories', fontsize = 18, fontweight = 'bold')
plt.title('Movie Genre', fontsize = 26, fontweight = 'bold', color = 'blue')
plt.xticks(rotation=80, fontsize = 13, fontweight = 'bold', color = 'green')
plt.yticks(fontsize = 12, fontweight = 'bold', color = 'green')
plt.show()


# In[34]:


df_combined = pd.concat([df_train, df_test], axis = 0)


# In[35]:


df_combined.head()


# In[37]:


df_combined.shape


# In[38]:


df_combined.size


# In[39]:


df_combined.isnull().any()


# In[40]:


df_combined.count()


# In[41]:


encoder = LabelEncoder()
df_combined["Category"] = encoder.fit_transform(df_combined["Category"].values)


# In[42]:


encoder = LabelEncoder()
df_combined["movie_name"] = encoder.fit_transform(df_combined["movie_name"].values)


# In[43]:


df_combined.head()


# In[45]:


df_combined.Category = df_combined.Category.fillna(df_combined.Category.mean())


# In[46]:


df_combined.count()


# In[47]:


df_combined.duplicated().values.any()


# In[48]:


vectorizer = TfidfVectorizer()


# In[49]:


X = vectorizer.fit_transform(df_combined["confession"])


# In[50]:


df_combined.head()


# In[51]:


y = df_combined["Category"]


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[53]:


naive_bayes_model = MultinomialNB()


# In[54]:


naive_bayes_model.fit(X_train, y_train)


# In[55]:


nb_predictions = naive_bayes_model.predict(X_test)


# In[56]:


print("Naive Bayes Model:")
print(confusion_matrix(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))
print("Accuracy: ", accuracy_score(y_test, nb_predictions))
print("r2_Score: ", r2_score(y_test, nb_predictions))


# In[ ]:


logistic_regression_model = LogisticRegression()


# In[59]:


logistic_regression_model.fit(X_train, y_train)


# In[61]:


lr_predictions = logistic_regression_model.predict(X_test)


# In[62]:


print("Logistic Regression Model:")
print(confusion_matrix(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))
print("Accuracy: ", accuracy_score(y_test, lr_predictions))
print("r2_Score: ", r2_score(y_test, lr_predictions))


# In[ ]:




