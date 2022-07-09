from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
import torch
import pickle
import numpy as np
import re

app = Flask(__name__)
model = pickle.load(open('LSTM_model.pkl', 'rb'))
words = pickle.load(open('words.pkl', 'rb'))
vocab2index = pickle.load(open('vocab2index.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


def tokenize(text):
    temp = re.sub('[^a-zA-Z]', ' ', text)
    temp = temp.lower()
    return [token for token in word_tokenize(temp)]


def encode_sentence(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        title = request.form['Title']
        author = request.form['Author']
        text = request.form['Text']
        nlp = title + author + text
        a = np.array(encode_sentence(nlp, vocab2index))
        X = a[0].reshape(-1, 70)
        X = torch.from_numpy(X.astype(np.int32))
        y_hat = model(X)
        pred = torch.max(y_hat, 1)[1]
        if pred == 0:
            return render_template('index.html', prediction_texts="The news article is possibly FAKE.")
        else:
            return render_template('index.html', prediction_text="The news article is possibly GENUINE.")
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
