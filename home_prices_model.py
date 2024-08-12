#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import streamlit as st


# Load the dataset
data = pd.read_csv("bhp.csv")

#Removing Outliers
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.Price_Per_sqft)
        st = np.std(subdf.Price_Per_sqft)
        reduced_df = subdf[(subdf.Price_Per_sqft>(m-st)) & (subdf.Price_Per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df1 = remove_pps_outliers(data)


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.Price_Per_sqft),
                'std': np.std(bhk_df.Price_Per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.Price_Per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df2 = remove_bhk_outliers(df1)


#Removing Column's
df3 = df2.drop(['size','Price_Per_sqft'],axis='columns')



# Encoding data
dummies = pd.get_dummies(df3.location)
df4 = pd.concat([df3,dummies.drop('other',axis='columns')],axis='columns')
#Drop location Column
df5 = df4.drop('location',axis='columns')



# Split the data into features (X) and target (y)
x = df5.drop(['price'],axis='columns')
y = df5.price


       
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Model
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(x_train,y_train)
lr_clf.score(x_test,y_test)

def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

#Pickel file
filename = 'final_Bagging_model.pkl'
#model 
pickle.dump(lr_clf, open(filename, 'wb'))
bag1.fit(x, y)
pk = lr_clf.predict(x_test)


Location = st.selectbox('location', data['location'].unique())
Total_sqft = st.selectbox('sqft', data['sqft'].unique())
Count_of_bathrooms = st.selectbox('bath', data['bath'].unique())
bhk = st.selectbox('bhk', data['bhk'].unique())


if st.button('Prevention Type'):
    df = {
        'location': Location,
        'sqft': Total_sqft,
        'bath': Count_of_bathrooms,
        'bhk': bhk,
    
    }

    df= pd.DataFrame(df5, index=[1])
    predictions = lr_clf.predict(df)

   

    st.title("Prize " + str(prediction_value))








