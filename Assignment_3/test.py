
# Required Imports
import pytest
import pickle
from score import score
import os
import requests
import time
import numpy as np
from urllib.parse import urlparse
import logging
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


spam_txt="subject : need outstanding logo ? working company ' image ? start visual identity key first good impression . help ! ' take part buildinq positive visual imaqe company creatinq outstandinq loqo , presentable stationery item professionai website . marketing toois significantly contributeto success business . take iook work sample , hot deal package see offer . work ! _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ interested . . . _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _"
ham_txt="subject : : spring 2001 conference participation jeffrey k . skilling vince , jeff decided decline speaking opportunity . scheduled speak leadership conference ut 2 / 16 , given time limitation , want pas one . sorry messenger bad news . sr vince j kaminski @ ect 10 / 12 / 2000 04 : 57 pm : sherri serum / corp / enron @ enron cc : vince j kaminski / hou / ect @ ect , richard causey / corp / enron @ enron subject : : spring 2001 conference participation jeffrey k . skilling sherri , resolution scheduling conflict jeff skilling february 22 nd ? friend ut ready make reservation send invitation conference vince"

#Define various functions for unittests using pytest


@pytest.fixture
def trained_model():
    # Load the best model saved during experiments
    # model = joblib.load('path/to/best_model.joblib')
    model ='logistic_regression'
    return model

def test_score_smoke(trained_model):
    # Smoke test
    text = "Sample text for testing"
    threshold = 0.5
    prediction, propensity = score(text, trained_model, threshold)
    assert prediction in [0, 1]
    assert 0 <= propensity <= 1

def test_score_format(trained_model):
    # Format test
    text = "Sample text for testing"
    threshold = 0.5
    prediction, propensity = score(text, trained_model, threshold)
    assert isinstance(prediction, int)
    assert isinstance(propensity, float)

def test_score_threshold_0(trained_model):
    # If threshold is 0, prediction should always be 1
    text = "Sample text for testing"
    threshold = 0
    prediction, _ = score(text, trained_model, threshold)
    assert prediction == 1

def test_score_threshold_1(trained_model):
    # If threshold is 1, prediction should always be 0
    text = "Sample text for testing"
    threshold = 1
    prediction, _ = score(text, trained_model, threshold)
    assert prediction == 0

def test_score_obvious_spam(trained_model):
    # On obvious spam input text, prediction should be 1
    text = "Buy cheap viagra now!!!"
    threshold = 0.5
    prediction, _ = score(text, trained_model, threshold)
    assert prediction == 1

def test_score_obvious_non_spam(trained_model):
    # On obvious non-spam input text, prediction should be 0
    text = "This is a legitimate email."
    threshold = 0.5
    prediction, _ = score(text, trained_model, threshold)
    assert prediction == 0


# Integration Test
def test_flask():
    # Launch the Flask app using os.system
    os.system('start /b python app_2.py')

    # Wait for the app to start up
    time.sleep(10)

    # Make a request to the endpoint
    response = requests.get('http://127.0.0.1:5000/')
    print(response.status_code)

    # Assert that the response is what we expect
    assert response.status_code == 200

    assert type(response.text) == str

    # Shut down the Flask app using os.system
    os.system('kill $(lsof -t -i:5000)')



# Run tests using pytest
# coverage.txt will be generated during the test run
# Use the following command to generate coverage.txt:
# pytest --cov-report term-missing --cov=test.py > coverage.txt

# Use the following command to generate and view test reports in html format:
# pytest test.py --html=report.html --verbose 
# start report.html
