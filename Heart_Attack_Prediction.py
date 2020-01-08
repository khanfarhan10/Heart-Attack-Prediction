#!/usr/bin/env python
# coding: utf-8

# In[362]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[363]:


df = pd.read_csv('data.csv')


# In[364]:


df.head()
df.rename(columns = {'num       ':'Target'}, inplace = True)


# In[365]:


df.drop(['slope','ca','thal'],axis=1,inplace=True)
df


# ## Number of missing values for each column

# In[366]:


columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']


# In[367]:


counts = []
for col in columns:
    counts.append(df[df[col]=='?']['age'].count())
counts


# In[368]:


missing_values = dict(zip(columns, counts))
missing_values


# ## Replace missing values in 'trestbps', 'chol', 'thalach' columns with mean of column

# In[369]:


columns_replaceby_mean = ['trestbps','chol','thalach']


# In[370]:


for item1 in columns_replaceby_mean:
    list1 = df[item1].tolist()
    list1 = filter(lambda a: a != '?', list1)
    list2 = []
    for item2 in list1:
        list2.append(float(item2))
    avg = np.average(list2)
    list3 = list(df[df[item1] == '?'].index)
    for item3 in list3:
        df[item1][item3] = avg


# In[371]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)


# ## Replace missing values in 'fbs', 'restecg', and 'exang' with most frequent value

# In[372]:


columns_replaceby_mode = ['fbs','restecg','exang']


# In[373]:


#all zero
for col in columns_replaceby_mode:
    print(df[col].mode())


# In[374]:


#replace '?' with 0
for col in columns_replaceby_mode:
    for i in range(0,len(df[col])-1):
        if df[col][i] == '?':
            df[col][i] = 0


# In[375]:


#all '?' fields replaced properly
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)


# ## Convert all values to float

# In[376]:


for col in columns:
        df[col] = df[col].astype(float)


# In[377]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)


# ## Data Visualization

# In[378]:


#0 = female, 1 = male
sns.boxplot(x='sex',y='chol',data=df)


# In[379]:


#restecg
#Value 0: normal
#Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria 20 ekgmo (month of exercise ECG reading)
sns.boxplot(x='restecg',y='chol',data=df)


# In[380]:


sns.heatmap(df.corr(),cmap='Blues')


# In[381]:


#0 = female, 1 = male
g = sns.FacetGrid(col='sex',data=df)
g = g.map(plt.hist, 'age')


# In[382]:


#Target: diagnosis of heart disease (angiographic disease status)
#Value 0: < 50% diameter narrowing
#Value 1: > 50% diameter narrowing
sns.lmplot(x='chol',y='trestbps',data=df,hue='Target',fit_reg=False)


# In[383]:


sns.lmplot(x='chol',y='thalach',data=df,hue='Target',fit_reg=False)


# In[384]:


sns.lmplot(x='thalach',y='trestbps',data=df,hue='Target',fit_reg=False)


# ## KNN Model

# In[385]:


from sklearn.preprocessing import StandardScaler


# In[386]:


scaler = StandardScaler()


# In[387]:


scaler.fit(df.drop('Target', axis=1))


# In[388]:


scaled_features = scaler.transform(df.drop('Target', axis=1))


# In[389]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[390]:


X = df_feat
y = df['Target']


# In[391]:


from sklearn.model_selection import train_test_split


# In[392]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)


# In[393]:


from sklearn.neighbors import KNeighborsClassifier


# In[394]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[395]:


knn.fit(X_train, y_train)


# In[396]:


predictions = knn.predict(X_test)


# In[397]:


from sklearn.metrics import classification_report,confusion_matrix


# In[398]:


print(confusion_matrix(y_test, predictions))


# In[399]:


print(classification_report(y_test,predictions))


# In[400]:


error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(y_test != pred_i))


# In[401]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',marker='o',mfc='pink',mec='red')
plt.title('Error Report for variable K')
plt.xlabel('K')
plt.ylabel('Error')


# In[442]:


knn = KNeighborsClassifier(n_neighbors=22)
knn.fit(X_train, y_train)
pred_22 = knn.predict(X_test)
print('K=22 (best model)')
print('\n')
print(confusion_matrix(y_test,pred_22))
print('\n')
print(classification_report(y_test,pred_22))


# ## Decision Tree

# In[403]:


from sklearn.model_selection import train_test_split


# In[404]:


X1=df.drop('Target',axis=1)
y1=df['Target']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3,random_state=3)


# In[405]:


from sklearn.tree import DecisionTreeClassifier


# In[406]:


dtree = DecisionTreeClassifier()


# In[407]:


dtree.fit(X1_train,y1_train)


# In[408]:


predictions1 = dtree.predict(X1_test)


# In[409]:


from sklearn.metrics import confusion_matrix, classification_report


# In[410]:


print(confusion_matrix(y1_test,predictions1))


# In[411]:


print(classification_report(y1_test,predictions1))


# ## Random Forest

# In[412]:


from sklearn.ensemble import RandomForestClassifier


# In[413]:


rForest = RandomForestClassifier()


# In[414]:


rForest.fit(X1_train,y1_train)


# In[415]:


rForest_predictions = rForest.predict(X1_test)


# In[416]:


print(classification_report(y1_test,rForest_predictions))


# In[417]:


print(confusion_matrix(y1_test,rForest_predictions))

