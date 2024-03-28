# Common imports
import numpy as np
import os
from urllib.parse import urlparse
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics._plot.precision_recall_curve import precision_recall_curve
from sklearn.metrics import auc
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import warnings
warnings.filterwarnings("ignore")

#Read the Data
train = pd.read_csv("./data/train.csv")
val = pd.read_csv("./Data/validation.csv")
test = pd.read_csv("./Data/test.csv")


#splitting the datframe into X and y
X_train,y_train = train["text"], train["spam"]
X_val,y_val = val["text"], val["spam"]
X_test,y_test = test["text"], test["spam"]

# Define function to fit classification models
def fit_model(train_data, y_train, model_name='logistic_regression'):
    if model_name == 'logistic_regression':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', LogisticRegression(random_state=42))
        ])
    elif model_name == 'random_forest':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', RandomForestClassifier(random_state=42))
        ])
    elif model_name == 'svm':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', SVC(probability=True))
        ])
    else:
        raise ValueError("Model name not recognized. Choose 'logistic_regression', 'random_forest', or 'svm'")

    model.fit(train_data, y_train)
    return model

# Define score function
def score(text:str, model_name, threshold:float) -> tuple:

    Model=fit_model(X_train, y_train, model_name)

    propensity = Model.predict_proba([text])[0][1]
    
    spam=0
    if float(propensity) >= threshold:
        spam=1
    else:
        spam=0
    
    return (spam, float(propensity))

