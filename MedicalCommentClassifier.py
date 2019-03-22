# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:12:25 2019

@author: carso
"""

import pandas as pd
import nltk
import os

def data2df (path, label):
    file, text = [], []
    for f in os.listdir(path):
        file.append(f)
        fhr = open(path+f, 'r', encoding='utf-8', errors='ignore') 
        t = fhr.read()
        text.append(t)
        fhr.close()
    return(pd.DataFrame({'file': file, 'text': text, 'class':label}))

dfneg = data2df('./NonPro/', 0) # NEG
dfpos = data2df('./Pro/', 1) # POS

df = pd.concat([dfpos, dfneg], axis=0)
df.sample(frac=0.005)
df.count() # check all files were read into df

X, y = df['text'], df['class']
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

import re
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
def preprocess(text):
    regex = re.compile(r"\s+")                               
    text = regex.sub(' ', text)    
    text = text.lower()          
    regex = re.compile(r"[%s%s]" % (string.punctuation, string.digits))
    text = regex.sub(' ', text)           
    sw = stopwords.words('english')
    text = text.split()                                              
    text = ' '.join([w for w in text if w not in sw]) 
    ' '.join([w for w in text.split() if len(w) >= 2])
    ps = PorterStemmer()
    text = ' '.join([ps.stem(w) for w in text.split()])
    return text

#build Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
clf = Pipeline(steps=[
    ('tfid', TfidfVectorizer(
        preprocessor=preprocess,
        lowercase=True, stop_words='english', 
        use_idf=True, smooth_idf=True, norm='l2',
        min_df=1, max_df=1.0, max_features=None, 
        ngram_range=(1, 1), sublinear_tf=True)),
    ('mdl',MultinomialNB())])
    
# setup grid search
from sklearn.model_selection import GridSearchCV
param_grid = {
    'mdl__alpha':[0.01, 0.1, 0.2, 0.5, 1],
    
    'tfid__norm':['l1', 'l2', None]
}
gscv = GridSearchCV(clf, param_grid, iid=False, cv=5, return_train_score=False)
# search for best parameters/estimator
gscv.fit(Xtrain, ytrain)
print(gscv.best_score_, "\n")
print(gscv.best_params_, "\n")

# evaluate best_estimator_ on test data
ypred = gscv.best_estimator_.predict(Xtest)
from sklearn import metrics
print (metrics.accuracy_score(ytest, ypred))
print (metrics.confusion_matrix(ytest, ypred))
print (metrics.classification_report(ytest, ypred))

#Voting Classifier
clf = Pipeline(steps=[
    ('tfid', TfidfVectorizer(
        preprocessor=preprocess,
        lowercase=True, stop_words='english', 
        use_idf=True, smooth_idf=True, norm='l2',
        min_df=1, max_df=1.0, max_features=None, 
        ngram_range=(1, 1), sublinear_tf=True)),
    ('vc', VotingClassifier(estimators=[('mdl', BaggingClassifier(MultinomialNB(alpha=.1),max_features=.5,max_samples=.5)),
    ('rf', RandomForestClassifier(n_estimators=100))]))
    ])

clf.fit(Xtrain,ytrain)
ypred = clf.predict(Xtest)
print (metrics.accuracy_score(ytest, ypred))
print (metrics.confusion_matrix(ytest, ypred))
print (metrics.classification_report(ytest, ypred))