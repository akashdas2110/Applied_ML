## Required Imports
from flask import Flask, request, render_template, url_for, redirect
import pickle
from score import score
import os
import pandas as pd

app = Flask(__name__,template_folder='template')
threshold=0.5


@app.route('/') 
def home():
    return render_template('input.html')


@app.route('/report', methods=['POST'])
def spam():
    sent = request.form['sent']
    label,prop=score(sent,"logistic_regression",threshold)
    lbl="Spam" if label == 1 else "Not spam"
    ans1 = f"""Iserted text is {sent}"""
    ans2 = f"""Prediction is {lbl}""" 
    ans3 = f"""Propensity score is {prop}"""
    return render_template('output.html', ans1 = ans1, ans2 = ans2, ans3 = ans3)


if __name__ == '__main__': 
    app.run(debug=True)
