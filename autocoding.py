#Autocoding EDA Grant Type

from __future__ import print_function
import scipy
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.externals import joblib

#Uploading the data
df = pd.read_excel ("G:/Shared/Disaster_RECOVERY_Assistance/COVID-19/Ryan's Working Files/Forecasting Project/FY_20_grants.xlsx", converters={'Project_Description': lambda x: str(x)})

#our training data set
df_training = df[df['Award.Month'] <= 8]

#our validation data set
df_validation = df[df['Award.Month'] >= 9]

vectorizer = CountVectorizer()
vectorizer.fit = df_training['Project_Description']

project_training = vectorizer.transform(df_training['Project_Description'])

description_vectorizer = CountVectorizer(df=5, ngram_range=(1,2))
description_vectorizer.fit(df_training['Project_Description'])

funding_vectorizer = CountVectorizer()
description_vectorizer.fit(df_training['EDA_Funding'])

x_description_training = description_vectorizer.transform(df_training['Project_Description'])
x_funding_training = funding_vectorizer.transform(df_training['EDA_Funding'])

x_training = scipy.sparse.hstack((x_description_training, x_funding_training))

y_training = df_training['EDA Program']
clf = LogisticRegression(C=2)
clf.fit(x_training, y_training)
