# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import json
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)


import pickle

with open('enc.pickle', 'wb') as file:
    pickle.dump(enc, file, pickle.HIGHEST_PROTOCOL)

# Load the model
model = pickle.load(open(r'C:\Users\zabbix_automation\PycharmProjects\AML_Assignment_1\model.pkl','rb'))

@app.route('/crop_prediction',methods=['POST'])
def crop_prediction():

    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(data['input'])

    # Take the first value of prediction
    output = prediction[0]

    print(output)

    return json.dumps(str(output))

if __name__ == '__main__':
    app.run(port=5000, debug=True)