import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result',methods=['POST'])
def result():
    sepal_length=float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    int_features=[sepal_length, sepal_width, petal_length, petal_width]
    final_features = [np.array(int_features)]
    prediction = str(model.predict(final_features))
    pred=prediction[2:-2]
    return render_template('index.html', pred_text='It is {} flower'.format(pred))


if __name__=="__main__":
    app.run(debug=True)

