#!/usr/bin/env python
# coding: utf-8

# # Name: Koer Pratiksha Jayant

# # Task3: House Price Prediction

# # 

# # EDA and Visualization

# In[1]:


##Import Libraries:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


##Load Dataset:

data = kc_house_data.csv
print(data.columns)
# Display the first few rows to understand the data
data.head()


# # Data Preprocessing

# In[3]:


data.shape


# In[4]:


##To check null value

data.isnull().sum()


# There are no missing values detected from the dataset

# In[5]:


##To check duplicated value

data.duplicated().sum()


# There are no duplicated value 

# In[6]:


##To check data type

data.dtypes


# In[7]:


##To Describe the dataframe

data.describe().style.background_gradient()


# In[8]:


data['datetime'] = pd.to_datetime(data['date'])
data['datetime']

data['year'] = data['datetime'].dt.year
data['year']

data['month'] = data['datetime'].dt.month
data['month']

data.drop(["id", "date", "datetime"], inplace=True, axis=1)
data     


# # 

# # Data Visualization

# In[9]:


plt.figure(figsize=(4, 4))
sns.heatmap(data.isna())
plt.show()


# In[10]:


largest_bedrooms = max(data['bedrooms'])
print("The largest number of bedrooms in the dataset =", largest_bedrooms)


# In[11]:


oldest_house = data['yr_built'].min()
newest_house = data['yr_built'].max()

print('The Oldest House:', oldest_house)
print('The Newest House:', newest_house)


# In[12]:


unique_years = data['yr_built'].unique()
print("Unique construction years:", unique_years)


# In[13]:


data.columns


# In[14]:


##To Visualize correlations

correlation = data.corr()
plt.figure(figsize=(20, 18))
sns.heatmap(correlation, linewidths=0.25, vmax=1.0, square=True, annot=True, cmap="cubehelix", linecolor='k')



# # 

# In[15]:


##To Create box plots
columns = ["bedrooms", "floors", "bathrooms", "grade"]
colors = ['gray', 'lightblue', 'red', 'blue']
fig, axes = plt.subplots(2, 2, figsize=(20, 18))
for i, column in enumerate(columns):
    ax = axes[i // 2, i % 2]
    sns.boxplot(data=data, x=data[column], y=data["price"], hue=None, color=colors[i], ax=ax)
    ax.set_title(f"Price vs {column}")


# # 

# Data Cleaning: Remove outliers based on Z-scores

# In[16]:


columns = data.columns.tolist()
outliers = []
threshold = 3
for col in columns:
    mean = np.mean(data[col])
    std = np.std(data[col])
    z_scores = (data[col] - mean) / std
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    outliers.extend(outlier_indices)
outliers = set(outliers)
outliers = list(outliers)


# Ratio of outliers present in dataset

# In[17]:


total_data_points = len(data)
total_outliers = len(outliers)
outlier_ratio = total_outliers / total_data_points

print("Ratio of outliers:", outlier_ratio)


# In[18]:


##To Remove rows with outliers from the DataFrame
data.drop(data.index[outliers], inplace=True)

##To Display the shape of the DataFrame after removing outliers
print("Shape of data after removing outliers:", data.shape)



# # 

#  Boxplot after removing the outliers

# In[19]:


columns = ["bedrooms", "floors", "bathrooms", "grade"]
colors = ['gray', 'lightblue', 'red', 'blue']
fig, axes = plt.subplots(2, 2, figsize=(20,18))
for i, column in enumerate(columns):
    ax = axes[i // 2, i % 2] 
    sns.boxplot(data=data, x=data[column], y=data["price"], hue=None, color=colors[i], ax=ax)
    ax.set_title(f"Price vs {column}")
plt.show()


# # 

# # Split Data and Standardize

# Split data into training and testing sets, and standardize features

# In[20]:


##To create feature matrix 'x' by dropping the 'price' column
x = data.drop(columns=['price'])

##To create the target variable 'y' containing the 'price' column
y = data['price']


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


##To Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2, shuffle=True
)

##To Display the sizes of the training and testing sets, as well as the total data size
train_size = x_train.shape[0]
test_size = x_test.shape[0]
total_size = data.shape[0]

print(f"Training data size: {train_size}")
print(f"Testing data size: {test_size}")
print(f"Total data size: {total_size}")


# In[23]:


##To print the shapes of the test and train datasets and their respective target variables
print("Test Data Shape:", x_test.shape)
print("Train Data Shape:", x_train.shape)
print("Target Train Shape:", y_train.shape)
print("Target Test Shape:", y_test.shape)


# In[24]:


from sklearn.preprocessing import StandardScaler


# In[25]:


##To create a StandardScaler object
scaler = StandardScaler()

##To fit and transform the training data
x_train_scaled = scaler.fit_transform(x_train)

##To transform the test data using the same scaler
x_test_scaled = scaler.transform(x_test)


# # 

# # Model Training and Model Evaluation

# In[26]:


##Importing all library need for model training

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from numpy.ma.core import floor
from sklearn.metrics import r2_score, mean_squared_error


# # Linear Regression

# In[27]:


lr = LinearRegression()
lr.fit(x_train, y_train)
pred_lr = lr.predict(x_test)


# In[28]:


r2_lr = r2_score(y_test, pred_lr)
print(f'Linear Regression R-squared: {r2_lr * 100:.2f}%')


# # SGD Regressor

# In[29]:


sgd = SGDRegressor()
sgd.fit(x_train, y_train)
pred_sgd = sgd.predict(x_test)


# In[30]:


mse_sgd = mean_squared_error(y_test, pred_sgd)
r2_sgd = r2_score(y_test, pred_sgd)

print(f'SGD Regressor R-squared: {r2_sgd * 100:.2f}%')
print(f'SGD Regressor Mean Squared Error: {mse_sgd:.2f}')


# # Random Forest Regressor

# In[31]:


rfg = RandomForestRegressor()
rfg.fit(x_train, y_train)
pred_rfg = rfg.predict(x_test)


# In[32]:


r2_rfg = rfg.score(x_test, y_test)
print(f'Random Forest Regressor R-squared: {r2_rfg * 100:.2f}%')


# # 

# # Prediction for a New House

# In[33]:


##To Create a DataFrame with feature names
feature_names = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above',
    'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat',
    'long', 'sqft_living15', 'sqft_lot15', 'year', 'month'
]


New_house = pd.DataFrame(data=np.array([
    [4, 2.00, 1920, 7803, 1.0, 0, 0, 3, 7, 1080, 840, 1962, 0, 98155, 47.7580, -122.325, 1940, 8147, 2014, 12]
]), columns=feature_names)

predicted_price = rfg.predict(New_house)
print('NEW House Price:', predicted_price)


# # 

# # END

# # 

# In[ ]:




