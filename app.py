import pickle
from flask import Flask, request, jsonify, app, url_for, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel.pkl', 'rb'))
scalar=pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')
## we are using this as an api
@app.route('/predict_api', methods=['POST'])
## when I give the input as a json format I will give it in the json format which will be captured inside the json key.
## As soon as we hit the API as a post request, whatever the
## information is present inside the data, we are going to capture it
## with the help of request.json module, the data would be stored in data variable.

def predict_api():
    data=request.json['data']
    ## scaling.pkl will standardize the entire data.
    ## data would be in the key value pairs, 
    print(data)
    ## I will get the dictionary values.
    print(np.array(list(data.values())).reshape(1, -1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The house price prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)







