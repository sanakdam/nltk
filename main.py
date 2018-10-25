from flask import (
	Flask, render_template, request, url_for, redirect
)
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import json
import unicodedata
import sys

app = Flask(__name__)
stemmer = LancasterStemmer()

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')	

@app.route('/<result>', methods=['GET'])
def index(result):
    return render_template('index.html', result=result)	

@app.route('/classify', methods=['POST'])
def classify():
	text = request.form['text']
	data = None
	words = []
	docs = []

	with open('data_json.json') as json_data:
		data = json.load(json_data)
	categories = list(data.keys())

	tbl = dict.fromkeys(i for i in range(sys.maxunicode)
		if unicodedata.category(u'i').startswith('P'))

	for each_category in data.keys():
	    for each_sentence in data[each_category]:
	        # remove any punctuation from the sentence
	        each_sentence = each_sentence.translate(tbl)
	#         print(each_sentence)
	        # extract words from each sentence and append to the word list
	        w = nltk.word_tokenize(each_sentence)
	#         print("tokenized words: ", w)
	        words.extend(w)
	        docs.append((w, each_category))

	# # stem and lower each word and remove duplicates
	words = [stemmer.stem(w.lower()) for w in words]
	words = sorted(list(set(words)))

	# reset underlying graph data
	tf.reset_default_graph()
	# Build neural network
	net = tflearn.input_data(shape=[None, 14045])
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, 9, activation='softmax')
	net = tflearn.regression(net)

	# Define model and setup tensorboard
	model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

	model.load('./model.tflearn')

	sentence_words = nltk.word_tokenize(text)
	sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
	bow = [0]*len(words)
	for s in sentence_words:
		for i, w in enumerate(words):
			if w == s:
				bow[i] = 1
	bows = np.array(bow)

	result = categories[np.argmax(model.predict([bows]))]
	print(result)

	return redirect(url_for('index', result=result))
   
if __name__ == "__main__":
	app.run(debug=True)