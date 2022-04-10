import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    Y = request.form.values()
    prediction = model.predict(Y)

    return render_template('index.html', {'prediction': prediction})


if __name__ == "__main__":
    app.run(debug=True)
