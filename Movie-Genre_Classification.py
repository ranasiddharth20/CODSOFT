


pwd




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





df_test = pd.read_csv('C:/Users/siddh/Desktop/Genre Classification Dataset/test_data.txt', sep=":::", header = 0 , engine = 'python')
df_train = pd.read_csv('C:/Users/siddh/Desktop/Genre Classification Dataset/train_data.txt',sep=":::", header = 0 , engine = 'python')
df_train.columns = ['Serial Number', 'movie_name', 'Category', 'confession']
df_test.columns = ['Serial Number', 'movie_name' , 'confession']





df_test.head()





df_train.head()





df_test.info()





df_train.info()





df_train.describe()





df_test.describe()





df_test.isnull().sum()





df_train.isnull().sum()





df_train.count()





df_test.count()





df_train.iloc[0:3]





df_test.shape





df_train.shape





plt.figure(figsize=(14,10))
sns.countplot(x='Category', data=df_train)
plt.xlabel('Movie Category')
plt.ylabel('Count')
plt.title('Movie Genre')
plt.xticks(rotation=90);
plt.show()



plt.xticks(rotation=80);





sns.displot(df_train.Category, kde=False, color = "grey")
plt.xticks(rotation=80);




plt.figure(figsize = (14,10))
count1 = df_train.Category.value_counts()
sns.barplot(x = count1, y = count1.index, orient = 'h')
plt.xlabel('Count')
plt.ylabel('Categories')
plt.title('Movie Genre')
plt.xticks(rotation=80)
plt.show()





plt.figure(figsize = (14,10))
count1 = df_train.Category.value_counts()
sns.barplot(x = count1, y = count1.index, orient = 'h')
plt.xlabel('Count', fontsize = 18, fontweight = 'bold')
plt.ylabel('Categories', fontsize = 18, fontweight = 'bold')
plt.title('Movie Genre', fontsize = 26, fontweight = 'bold', color = 'blue')
plt.xticks(rotation=80, fontsize = 13, fontweight = 'bold', color = 'green')
plt.yticks(fontsize = 12, fontweight = 'bold', color = 'green')
plt.show()





df_combined = pd.concat([df_train, df_test], axis = 0)




df_combined.head()





df_combined.shape





df_combined.size




df_combined.isnull().any()


# In[40]:


df_combined.count()




encoder = LabelEncoder()
df_combined["Category"] = encoder.fit_transform(df_combined["Category"].values)





encoder = LabelEncoder()
df_combined["movie_name"] = encoder.fit_transform(df_combined["movie_name"].values)





df_combined.head()





df_combined.Category = df_combined.Category.fillna(df_combined.Category.mean())





df_combined.count()





df_combined.duplicated().values.any()





vectorizer = TfidfVectorizer()




X = vectorizer.fit_transform(df_combined["confession"])





df_combined.head()





y = df_combined["Category"]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





naive_bayes_model = MultinomialNB()





naive_bayes_model.fit(X_train, y_train)





nb_predictions = naive_bayes_model.predict(X_test)





print("Naive Bayes Model:")
print(confusion_matrix(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))
print("Accuracy: ", accuracy_score(y_test, nb_predictions))
print("r2_Score: ", r2_score(y_test, nb_predictions))





logistic_regression_model = LogisticRegression()





logistic_regression_model.fit(X_train, y_train)





lr_predictions = logistic_regression_model.predict(X_test)

print("Logistic Regression Model:")
print(confusion_matrix(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))
print("Accuracy: ", accuracy_score(y_test, lr_predictions))
print("r2_Score: ", r2_score(y_test, lr_predictions))







