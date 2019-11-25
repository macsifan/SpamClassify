from flask import Flask, request, jsonify, render_template
import pickle
import utils
import numpy as np
import re

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def processEmail(email_contents, verbose=True):
    vocabList = utils.getVocabList()
    word_indices = []
    email_contents = email_contents.lower()
    email_contents = re.compile('<[^<>]+>').sub(' ', email_contents)
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)
    email_contents = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', email_contents)
    email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)
    email_contents = [word for word in email_contents if len(word) > 0]
    stemmer = utils.PorterStemmer()
    processed_email = []
    for word in email_contents:
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)
        processed_email.append(word)
        if len(word) < 1:
            continue
        if word in vocabList:
            word_indices.append(vocabList.index(word))
    return word_indices

def emailFeatures(word_indices):
    n = 1899
    x = np.zeros(n)
    x[word_indices] = 1
    return x

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = ''.join(int_features)
    word_indices = processEmail(final_features, verbose=False)
    x = emailFeatures(word_indices)
    prediction = utils.svmPredict(model, x)
    output ='spam' if prediction else 'not spam'
    return render_template('index.html', prediction_text='This email is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    final_features=''.join(list(data.values()))
    word_indices = processEmail(final_features, verbose=False)
    x = emailFeatures(word_indices)
    prediction = utils.svmPredict(model,x)
    output = 'spam' if prediction else 'not spam'
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)



