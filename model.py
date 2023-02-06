import keras
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.preprocessing import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
import pandas as pd
from keras.callbacks import ModelCheckpoint
import numpy as np
from prediction import predict

from keras.utils import np_utils
import pickle

max_sequence_len = 88
model = keras.models.load_model('my_model.h5')
with open('notebooks/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def make_lyrics(seed_text, next_words):
    pred_index=[]
    for i in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list],
                     maxlen=max_sequence_len-1,padding='pre')
        #print(token_list.shape)
        token_list = np.reshape(token_list, (1, max_sequence_len-1, 1))
        predicted = model.predict(token_list, verbose=0)
        predicted_index =  np.argmax(predicted)
        pred_index.append(predicted_index)



        #predicted_index=1
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)
    return seed_text

make_lyrics('I will', 10)
