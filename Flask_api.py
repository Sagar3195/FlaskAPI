from flask import *
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_pin = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_pin)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict', methods = ['GET'])
def predict_note_authintication():
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
        - name : variance
          in : query
          type : number
          required : true
        - name : skewness
          in : query
          type : number
          required : true
        - name : curtosis
          in : query
          type : number
          required : true
        - name : entropy
          in : query
          type : number
          required : true
    responses:
        200:
            description: The output values

    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)

    return "The predicted value is " + str(prediction)

@app.route("/predict_file", methods = ['POST'])
def predict_file():
    """Let's Authentication the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true

    responses:
        200:
            description : The output values

    """
    data = pd.read_csv(request.files.get('file'))
    prediction_test = classifier.predict(data)
    return "The predicted value for test data is  " + str(list(prediction_test))


if __name__ == "__main__":
    app.run(debug= True)






