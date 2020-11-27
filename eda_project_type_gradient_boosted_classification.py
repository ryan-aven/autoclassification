# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:11:25 2020

@author: RAven
"""
print("importing base libraries...")
import pandas as pd
import re
import nltk
print("base libraries imported")

#importing the data set
print("reading in Data...")
df = pd.read_excel ("G:/Shared/Disaster_RECOVERY_Assistance/COVID-19/Ryan's Working Files/Forecasting Project/FY_20_grants.xlsx", converters={'Project_Description': lambda x: str(x)})
print("data loaded successfully")

#setting x and y values
y = df.EDA_Program
x = df.Project_Description

#doing some document editing to remove special characters
print("removing special characters and lemmatization...")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')

documents = []

stemmer = WordNetLemmatizer()

for sen in range(0, len(x)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(x[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)
print("special characters removed, lemmatization complete")

#converting text to numbers
print("applying vectorizer...")
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=11, max_df=.6, stop_words=stopwords.words('english'))
x = vectorizer.fit_transform(documents).toarray()

#reweighting based on tf-idf values
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
x = tfidfconverter.fit_transform(x).toarray()

#splitting the data sets up into training and test data sets
print("splitting data sets into testing and training sets")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#importing the gradient boosted classifier
print("uploading gradient boosted classifier")
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(max_depth=8, learning_rate=0.1, random_state=0)
classifier.fit(x_train, y_train)

#making our predictions
print("fitting model...")
y_pred = classifier.predict(x_test)

#producing summary metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
predict_log_proba(x_test)
