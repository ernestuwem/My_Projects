#!/usr/bin/env python
# coding: utf-8

# # <h1 style="font-family: Trebuchet MS; padding: 20px; font-size: 40px; color: #4B94FF; text-align: center; line-height: 0.55;background-color: #000000"><b>Connecttel Customer Churn Prediction</b><br></h1>
# 

# 
# <center>
#     <img src="https://camo.githubusercontent.com/c098ac6a4a68b17721954455b480fd944339792580f7958bbd8a6f8d2dd83c6b/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f76322f726573697a653a6669743a3836302f302a7354482d334d505a56636961573364552e6a706567" alt="Connectell Customer Churn" width="100%">
# </center>
# 
# ### Problem Statement :
# 
# ConnectTel Telecom Company faces the challenge of customer churn, which threatens its sustainability and growth in the telecommunications industry. The current retention strategies lack effectiveness, resulting in lost customers to competitors. To address this, ConnectTel aims to develop a customer churn prediction system using advanced analytics and machine learning. By analyzing customer data, the company seeks to identify potential churn indicators early and implement targeted retention initiatives. This proactive approach will allow ConnectTel to optimize resource allocation, enhance customer loyalty, and maintain a competitive edge. Overall, the churn prediction system represents a strategic initiative to mitigate customer attrition and foster long-term success for ConnectTel in the telecommunications market.
# 
# 
# 

# In[1]:


# Importation of Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Load the dataset


# In[3]:


df= pd.read_csv(r"C:\Users\erNEST.UDOUDO\Downloads\10alytics\10ALYTICS-PROJECT\Customer-Churn.csv")
df.head()


# # Reviewing the dataset for correctness
# 

# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


# Change TotalCharges object to float since Totalcharges are meant to be expressed as numeric variables


# In[8]:


df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()


# In[9]:


df.info()


# In[10]:


df["Churn"].value_counts()#counting the number of customers in the dataset who have churned: 


# # Exploratory Data Analysis for Conecttinel Customer Churn
# 

# In[11]:


# Analyzing the dataset with respect to gender, SeniorCitizen, Partner and Dependents

# Define colors for each category
colors = {'gender': ['pink', 'Darkblue'],         # Assuming 'gender' has two categories
          'SeniorCitizen': ['gray', 'brown'], # Assuming 'SeniorCitizen' has two categories
          'Partner': ['Lightgreen', 'Darkblue'],      # Assuming 'Partner' has two categories
          'Dependents': ['Lightblue', 'Lightgreen']}  # Assuming 'Dependents' has two categories

cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
numerical = cols

plt.figure(figsize=(20, 4))

for i, col in enumerate(numerical):
    ax = plt.subplot(1, len(numerical), i+1)
    sns.countplot(x=str(col), data=df, palette=colors[col])  # Use palette argument to specify colors
    ax.set_title(f"{col}")

plt.show()


# Most customers in the dataset are younger individuals without a dependent. There is an equal distribution of user gender and marital status.

# In[12]:


# Checking the relationship between Churn and monthlyCharges
plt.figure(figsize=(8, 6))

sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette=['Lightblue', 'darkblue'])

plt.title('Monthly Charges by Churn Status')
plt.xlabel('Churn')
plt.ylabel('Monthly Charges')

plt.show()


# The assumption above is true. Customers who churned have a higher median monthly charge than customers who renewed their subscription.

# In[13]:


# Comparing Churn with other variables like Internetservice,Techsupport,Onnlinebackup and contrcats 

colors = ['darkblue', 'lightblue', 'blue']

cols = ['InternetService', 'TechSupport', 'OnlineBackup', 'Contract']

plt.figure(figsize=(14, 4))

for i, col in enumerate(cols):
    ax = plt.subplot(1, len(cols), i+1)
    sns.countplot(x="Churn", hue=str(col), data=df, palette=colors)
    ax.set_title(f"{col}")


# The analysis identifies key factors contributing to customer churn:
# 
# 1.InternetService: Customers using fiber optic Internet are more likely to churn, possibly due to high costs or inadequate coverage.
# 
# 2.TechSupport: Many churned users didn't utilize tech support, suggesting unresolved technical issues as a churn driver.
# 
# 3.OnlineBackup: Churned customers often lacked online backup services, indicating a need for better data protection measures.
# 
# 4.Contract Type: Churn was prevalent among users on monthly contracts, likely due to the ease of cancellation.
# 
# Data-driven insights are crucial for understanding churn patterns. Addressing issues such as the lack of tech support can inform targeted retention strategies, like offering complimentary support services. Ultimately, businesses must proactively tackle churn to enhance customer satisfaction and long-term profitability. 

# # Encoding Categorical Variables
# 
# The categorical variables in the dataset need to be converted into a numeric format before we can feed them into the machine learning model. We will perform the encoding using Scikit-Learn’s label encoder.
# 
# First, let’s take a look at the categorical features in the dataset:

# In[14]:


cat_features = df.drop(['customerID','TotalCharges','MonthlyCharges','SeniorCitizen','tenure'],axis=1)

cat_features.head()


# In[15]:


#Running the below code will make all the categorical values in the dataset to numbers

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
df_cat = cat_features.apply(le.fit_transform)
df_cat.head()


# In[16]:


# Merging the dataframe with the previous one we created.

num_features = df[['customerID','TotalCharges','MonthlyCharges','SeniorCitizen','tenure']]
finaldf = pd.merge(num_features, df_cat, left_index=True, right_index=True)


# # Building the Customer Churn Prediction Model
# 
# 

# In[23]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=46)
rf.fit(X_train,y_train)


# # Customer Churn Prediction Model Evaluation
# Evaluation of the model predictions on the test dataset

# In[25]:


from sklearn.metrics import accuracy_score

preds = rf.predict(X_test)
print(accuracy_score(preds,y_test))


# # Our model is performing well, with an accuracy of approximately 0.78 on the test dataset.

# In[ ]:




