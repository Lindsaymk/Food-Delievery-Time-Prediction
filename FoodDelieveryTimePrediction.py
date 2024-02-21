#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd


# In[7]:


import numpy as np


# In[8]:


import plotly.express as px


# In[9]:


#upload the dataset
file_path = r'C:\Users\Hp\Desktop\deliverytime.txt'
data = pd.read_csv(file_path, delimiter=',')


# In[10]:


print(data.head())


# In[11]:


data.info()


# In[12]:


#checking for null values
data.isnull().sum() 


# In[13]:


#checking for missing values
data.isna().sum()


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[17]:


# Prepare the features (X) and target variable (y)
X = encoded_data.drop(columns=['Time_taken(min)'])  # Features
y = encoded_data['Time_taken(min)']  # Target variable


# In[19]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


data.info()


# In[16]:


encoded_data = pd.get_dummies(data, columns=['Delivery_person_ID', 'Delivery_person_Age', 'Type_of_order', 'Type_of_vehicle', 'ID'])


# In[27]:


data['Delivery_person_ID'] = data['Delivery_person_ID'].astype(str)
data['Delivery_person_Age'] = data['Delivery_person_Age'].astype(str)
data['Type_of_order'] = data['Type_of_order'].astype(str)
data['Type_of_vehicle'] = data['Type_of_vehicle'].astype(str)


# In[18]:


encoded_data.info()


# In[21]:


print(encoded_data.columns)


# In[23]:


#reducing dimensionality
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_subset = encoded_data[['Delivery_person_Ratings', 'Restaurant_latitude',
                         'Restaurant_longitude', 'Delivery_location_latitude',
                         'Delivery_location_longitude','Time_taken(min)']]

y = encoded_data['Time_taken(min)']
X_subset = X_subset.drop(columns=['Time_taken(min)'])  # Drop the target variable

# Standardizing the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)

# Instantiating PCA and specifying the desired number of components or explained variance ratio
pca = PCA(n_components=0.95)  # Retain 95% of variance

# Applying PCA transformation to the standardized feature matrix
X_pca = pca.fit_transform(X_scaled)

# Checking the shape of the transformed feature matrix
print("Shape of X_pca:", X_pca.shape)


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model using Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)


# In[25]:


# Example input values
new_data = [[4.5, 40.7, -74.0, 40.8, -73.9]]  

# Standardizing the new input data using the same scaler object used for training
new_data_scaled = scaler.transform(new_data)

# Applying PCA transformation to the standardized new input data
new_data_pca = pca.transform(new_data_scaled)

# Making predictions using the trained model
predicted_delivery_time = model.predict(new_data_pca)

print("Predicted delivery time:", predicted_delivery_time)


# In[ ]:




