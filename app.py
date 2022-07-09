from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
import torch
import pickle
import numpy as np
import re
import torch.nn as nn
import nltk

nltk.data.path.append('./nltk_data/')
app = Flask(__name__)

words = pickle.load(open('words.pkl', 'rb'))
vocab2index = pickle.load(open('vocab2index.pkl', 'rb'))


class LSTM_fixed_len(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])

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
        if pred == 1:
            return render_template('index.html', prediction_text="The news article is possibly FAKE.")
        else:
            return render_template('index.html', prediction_text="The news article is possibly GENUINE.")
    else:
        return render_template('index.html')


if __name__ == "__main__":
    vocab_size = len(words)
    model = LSTM_fixed_len(vocab_size, 50, 50)
    model.load_state_dict(torch.load('LSTM_fixed.pt'))
    app.run(debug=True)
