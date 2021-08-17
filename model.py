import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import Input,layers,Model,preprocessing
from sklearn.model_selection import train_test_split
import pickle
import os

embedding_dim = 256
vocab_size = 99999
max_lengths = 300

df = pd.read_csv('./dataset/sample_dataset.csv')
df = df.dropna()
text_data = df['text_response']
mcqs_data = df.drop(['text_response','_id','job_family','gender'],axis=1)

def text_processing(text):
    tokenizer = preprocessing.text.Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(text)
    sequence = tokenizer.texts_to_sequences(text)
    padded_text = tf.keras.preprocessing.sequence.pad_sequences(sequence,truncating='post',maxlen=max_lengths)
    return padded_text,tokenizer

text_data,tokenizer = text_processing(np.array(text_data))
concat_ds = pd.concat([pd.DataFrame(text_data),mcqs_data],axis=1).dropna()
concat_ds['selected'] = concat_ds['selected'].replace(-1,2)


filename = 'tokenizer.h5'
output_path = os.path.join('./app/',filename)
outfile = open(output_path,'wb')
pickle.dump(tokenizer,outfile)
outfile.close()

X = concat_ds.drop('selected',axis=1).values
y = concat_ds['selected'].values
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=1)

inputs = Input(shape=(None,))
x = layers.Embedding(vocab_size,embedding_dim,input_length=max_lengths)(inputs)
x = layers.GRU(embedding_dim,return_sequences=True,activation='tanh')(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
x = layers.Flatten()(x)
x = layers.Dense(32,activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(3,activation='softmax')(x)
model = Model(inputs=inputs,outputs=x)
model.summary()

e_stop = tf.keras.callbacks.EarlyStopping(patience=3)
lr_decay = tf.keras.callbacks.ReduceLROnPlateau(patience=3)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['acc'])

model.fit(xtrain,ytrain,epochs=10
        ,validation_data=(xtest,ytest)
        ,batch_size=32
        ,callbacks=[e_stop,lr_decay])

model.evaluate(xtest,ytest)

tf.saved_model.save(model,'./app/')