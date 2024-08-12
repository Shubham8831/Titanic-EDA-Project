#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df=pd.read_csv("data.csv")


# In[6]:


df.head()


# In[7]:


#checking shape of data
df.shape


# In[8]:


#checking null values
df.isnull().sum()
#(we can clearly see in total 891 values
#cabin has missing 687 values so its of no use.)


# In[9]:


#checking duplicate values
df.duplicated().sum()


# In[10]:


# remove unwanted columns[passangerid, name, ticket, cabin]
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1,inplace = True)


# In[11]:


df.head(2)


# In[12]:


df.isnull().sum()


# In[13]:


df.dropna(inplace=True)


# In[14]:


df.shape


# In[15]:


df.info()


# In[16]:


df.describe() # only take numerical columns removed object columns


# In[17]:


df.columns


# In[18]:


#finding categorical or numerical data with value_counts() [survived is categorical data]
df["Survived"].value_counts()


# In[19]:


df["Age"].value_counts()
#age not categorical its numerical 


# In[20]:


#categorical ->[survived, pclass, gender, embarked,dibdp,parch]
#numerical ->[age,fare]


# In[21]:


#individual exploring each data


# In[22]:


#categorical data
#survived
df["Survived"].value_counts()


# In[23]:


plt.figure(figsize=(3,3))
sns.countplot(x="Survived",data=df)


# In[24]:


#pclass
df["Pclass"].value_counts()


# In[25]:


plt.figure(figsize=(3,3))
sns.countplot(x="Pclass",data=df)


# In[26]:


#gender
df["Gender"].value_counts()


# In[27]:


plt.figure(figsize=(3,3))
sns.countplot(x="Gender",data=df)


# In[28]:


#embarked
df["Embarked"].value_counts()


# In[31]:


plt.figure(figsize=(3,3))
sns.countplot(x="Embarked",data=df)


# In[32]:


df["SibSp"].value_counts()


# In[33]:


plt.figure(figsize=(3,3))
sns.countplot(x="SibSp",data=df)


# In[34]:


#parch
df["Parch"].value_counts()


# In[35]:


plt.figure(figsize=(3,3))
sns.countplot(x="Parch",data=df)


# In[36]:


#Numerical data
#age
df["Age"].describe()


# In[37]:


plt.figure(figsize=(4,4))
plt.hist(df["Age"])
plt.show()


# In[39]:


#fare
df["Fare"].describe()


# In[41]:


plt.figure(figsize=(4,4))
plt.hist(df["Fare"])
plt.show()


# ### advance analytics 

# In[42]:


#survived vs gender "both are categorica columns"
df.head()


# In[54]:


#so we will group the data
grouped_data=df.groupby(["Gender","Survived"]).size().unstack()


# In[52]:


grouped_data


# In[58]:


plt.figure(figsize=(3,3))
grouped_data.plot(kind="bar")
plt.title("Gender vs Survived")
plt.xlabel("Gender")
plt.ylabel("No. of People")
plt.show()


# In[62]:


#pclass vs survived
grouped_data1=df.groupby(["Pclass","Survived"]).size().unstack()


# In[67]:


plt.figure(figsize=(3,3))
grouped_data1.plot(kind="bar", stacked=True)
plt.title("Pclass vs Survived")
plt.xlabel("Pclass")
plt.ylabel("No. of people")
plt.show()


# In[68]:


#survived vs age[categorical vs numerical]
df[df["Survived"]==1]["Age"]


# In[73]:


sns.kdeplot(df[df["Survived"]==1]["Age"],color="green",label="survived")
sns.kdeplot(df[df["Survived"]==0]["Age"],color="red",label="not survived")
plt.legend()
plt.show()


# In[75]:


#embarked vs survived [both are categorical data]
df.head()


# In[79]:


grouped_data3 = df.groupby(["Embarked","Survived"]).size().unstack()


# In[80]:


grouped_data3


# In[84]:


plt.figure(figsize=(3,3))


# In[83]:


#survival vs fare [categorical vs numerical]


# In[85]:


sns.kdeplot(df[df["Survived"]==1]["Fare"],color="green",label="survived")
sns.kdeplot(df[df["Survived"]==0]["Fare"],color="red",label="not survived")
plt.legend()
plt.show()


# In[86]:


#correlation


# In[88]:


numeric_df = df.select_dtypes(include = ["float","int"])


# In[91]:


corr=numeric_df.corr()


# In[92]:


corr


# In[94]:


sns.heatmap(corr, cmap="coolwarm",annot=True)


# In[96]:


sns.pairplot(df)


# In[97]:


#age vs gender


# In[100]:


male_data = df[df["Gender"]=="male"]
female_data = df[df["Gender"]=="female"]


# In[106]:


sns.histplot(data=male_data,x="Age",color="blue",label="Male",alpha=0.5)
sns.histplot(data=female_data,x="Age",color="red",label="FeMale",alpha=0.5)
plt.legend()
plt.show()


# In[ ]:


#Mini Project on Titanic dataset
#tech used [python,numpy,pandas,eda]

