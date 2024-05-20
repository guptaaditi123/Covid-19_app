
import streamlit as st
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# loading dataset
df = pd.read_csv("Covid-19.csv")

df = df.drop(range(40001, 49999))

#viewing head of data set
#df.head()

# 1 is for positive
# 0 is for negative

df['POSITIVE'] = df['CLASIFFICATION_FINAL'].apply(lambda x: 1 if x <= 3 else 0)

df.drop(columns='CLASIFFICATION_FINAL', inplace=True)

df['POSITIVE'].value_counts()

#Here we will drop data as we do not need it
df=df.drop(['USMER'], axis=1)
df=df.drop(['PNEUMONIA'], axis=1)
df=df.drop(['PREGNANT'], axis=1)
df=df.drop(['DIABETES'], axis=1)
df=df.drop(['COPD'], axis=1)
df=df.drop(['ASTHMA'], axis=1)  # axis=1 to drop full column
df=df.drop(['HIPERTENSION'], axis=1)
df=df.drop(['OTHER_DISEASE'], axis=1)
df=df.drop(['CARDIOVASCULAR'],axis=1)

df.drop(df.index[df.INTUBED == 99], axis=0, inplace=True)
df.drop(df.index[df.INTUBED == 97], axis=0, inplace=True)
df.drop(df.index[df.ICU== 98], axis=0, inplace=True)

df['DIED'] = [0 if i=='9999-99-99' else 1 for i in df.DATE_DIED]

df=df.drop(['DATE_DIED'], axis=1)

# Chosing target feature
target = "DIED"

# Splitting data
X = df.drop(target, axis=1)
Y = df[target]

# Printing X
#X

#Y

from imblearn.under_sampling import RandomUnderSampler
sampler = RandomUnderSampler(random_state=42)
X_sampled, Y_sampled = sampler.fit_resample(X, Y)
Y_sampled.value_counts()

#Splitting to traon and test set
X_train, X_test, Y_train, Y_test= train_test_split(X_sampled, Y_sampled, test_size= 0.2, random_state=0)

#Applying GradientBoostingClassifier Algorithm
model=GradientBoostingClassifier()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

#Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy of training data: ',training_data_accuracy)

#Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy of test data: ',test_data_accuracy)

#website
# Web Title
st.title('COVID-19 Prediction/Classification')
# Split Columns
col1, col2 = st.columns(2)

with col1:
    MEDICAL_UNIT = st.text_input('Medical unit (0 OR 1)')
    PATIENT_TYPE = st.text_input('Patient type (1-returned home OR 2-hospitalization)')
    AGE = st.text_input('Age')
    ICU = st.text_input('ICU (1-yes or 2-no)')

with col2:
    SEX = st.text_input('Sex (1-female OR 2-male)')
    INTUBED = st.text_input('Ventilator (1-yes, 2-no)')
    INMSUPR = st.text_input('Inmsupr (1-yes OR 2-no)')
    POSITIVE = st.text_input('Positive (1-positive, 2-not positive)')
# Define vectorization

# Create a DataFrame with named features
input_data_df = pd.DataFrame({
    'MEDICAL_UNIT': [MEDICAL_UNIT],
    'SEX': [SEX],
    'PATIENT_TYPE': [PATIENT_TYPE],
    'INTUBED': [INTUBED],
    'AGE': [AGE],
    'INMSUPR': [INMSUPR],
    'ICU': [ICU],
    'POSITIVE': [POSITIVE]
})

def prediction(input_data_df):
    prediction=model.predict(input_data_df)
    return prediction[0]


if st.button('COVID-19 Prediction Test'):
    pred = prediction(input_data_df)
    if pred ==1:
        st.write('The person is dead')
    else:
        st.write('The person is recovered')
