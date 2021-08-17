from flask import request,jsonify, Flask
import tensorflow as tf
import pandas as pd
import numpy as np 
import pickle
import json

app = Flask(__name__)

model = tf.keras.models.load_model('./')
max_lengths = 300

filename = 'tokenizer.h5'
infile = open(filename,'rb')
tokenizer = pickle.load(infile)
infile.close()

def text_processing(text,tokenizer):
    tokenizer.fit_on_texts(text)
    sequence = tokenizer.texts_to_sequences(text)
    padded_text = tf.keras.preprocessing.sequence.pad_sequences(sequence,truncating='post',maxlen=max_lengths)
    return padded_text

def return_prediction(data):
    df = pd.DataFrame([data])
    text_data = df.iloc[0,:].values[-1]['text_response']
    df = list(df.values[0][0].values())[:-1]
    processed_data = text_processing([text_data],tokenizer)
    concat_ds = pd.concat([pd.DataFrame(processed_data[0]).T,pd.DataFrame(df).T], axis=1)
    result = model.predict(concat_ds)
    return result

@app.route("/")
def index():
    return '<h1> Endpoint is running</h1>'

@app.route("/api",methods=['POST'])
def api():
    content = request.get_json()
    result = return_prediction(content).tolist()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)