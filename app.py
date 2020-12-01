from flask import *
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

pickle_in = open("classifier.pkl", 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All!!!!"

@app.route('/predict', methods = ['GET'])
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)
    return "The predicted value is" + str(prediction)

@app.route('/predict_file', methods = ['POST'])
def predict_file():
    df_test = pd.read_csv(request.files.get('file'))
    prediction_file = classifier.predict(df_test)
    print(prediction_file)
    return "The predicted value for test data is " + str(list(prediction_file))


if __name__ == '__main__':
    app.run(debug= True)