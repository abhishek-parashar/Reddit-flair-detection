import pickle
import logging
import praw
from praw.models import MoreComments
import os
import flask
from flask import Flask, flash, request,jsonify, json
import json
import joblib
import numpy as np 
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('all')
from bs4 import BeautifulSoup


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
def text_cleaning(text):
    text = BeautifulSoup(text, "lxml").text
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def string(value):
    return str(value)


app = Flask(__name__,template_folder='templates')


# loading the model 
model = joblib.load(open('model/xgb.bin', 'rb'))

reddit = praw.Reddit(client_id='QPdCUgBcp4WinA', client_secret='HF-sKHVC5Os3gufVxWvzIKijNb4', user_agent='reddit-flair', username='reddit-flair', password='flair123')

# perdiction function
def prediction(url):
	submission = reddit.submission(url = url)
	data = {}
	data["title"] = str(submission.title)
	data["url"] = str(submission.url)
	data["body"] = str(submission.selftext)

	submission.comments.replace_more(limit=None)
	comment = ''
	count = 0
	for top_level_comment in submission.comments:
		comment = comment + ' ' + top_level_comment.body
		count+=1
		if(count > 10):
			 break

	data["comment"] = str(comment)

	data['title'] = text_cleaning(str(data['title']))
	data['body'] = text_cleaning(str(data['body']))
	data['comment'] = text_cleaning(str(data['comment']))

	combined_features = data["title"] + data["comment"] + data["body"] + data["url"]

	return model.predict([combined_features])

# initialising flask api

# Setting up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
	if flask.request.method == 'GET':
		# rendering for input
		return(flask.render_template('main.html'))

	if flask.request.method == 'POST':
		# Extracting  the input
		text = flask.request.form['url']

		# model's prediction
		flair = str(prediction(str(text)))

		# Rendering the form again and reminding of the previous value
		return flask.render_template('main.html', original_input={'url':str(text)}, result=flair[2:-2])



@app.route("/automated_testing",methods=['POST'])
def test():
	if request.files:
			file = request.files["upload_file"]
			texts = file.read()
			texts = str(texts.decode('utf-8'))
			links = texts.split('\n')
			pred = {}
			for link in links:
				pred[link] =  str(prediction(str(link)))[2:-2]
			return jsonify(pred)
	else:
			return 400

if __name__ == '__main__':
	app.run()