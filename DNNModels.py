# file loading and os packages
import os
from urllib.request import urlretrieve
from os.path import isfile, isdir, join
import shutil
from sklearn.metrics import f1_score, accuracy_score

# Basic packages
import pandas as pd
import numpy as np
import re
import collections
import matplotlib.pyplot as plt
from pathlib import Path
import os
from urllib.request import urlretrieve
from os.path import isfile, isdir, join
import shutil
import seaborn as sns

# Packages for data preparation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import initializers
from keras.optimizers import Adam
# Packages for modeling
from keras import models as Models
from keras.layers import *
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K


'''

Implementation of RNN many to one for text classification

params= {
    'embed_dim': 256,
    'NB_WORDS': NB_WORDS,
    'rnn_out': 64,
    'MAX_LEN': max_len,
    'iter': 1000,
    'lr':1e-3,
    'save_path': join(save_path, task+'model_1.hdf5'),
    'pretrain_emb':None,
    'batch_size': 32
    'y_type': binary
}


To generate np.array of y:
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y)
    # to convert your pred to text
    df.prediction = le.inverse_transform(np.argmax(dnn.predict(X))
    # le.inverse_transform(prediction)
    
To train:
dnn = emb_dnn(params)
dnn.train(X,y)
dnn.fine_tune(X,y)
predictions = dnn.predict(X)


'''

class emb_dnn():
    '''
    param: 'embed_dim': dimension of your embedding vector
           'NB_WORDS': number of words
           'rnn_out': output dim of rnn layer
           'MAX_LEN': max_len generate from above
           'iter': maximum iteration training
           'lr': initial learning rate
           'save_path': dir to save the model
           'pretrain_emb': optional pretrained w2v
           'batch_size': batch size
           'y_type' : binary or multi
    NB_WORDS: number of words of input words seqs
    '''

    def __init__(self, params):
        self.embed_dim = params['embed_dim']
        self.NB_WORDS = params['NB_WORDS']
        self.rnn_out = params['rnn_out']
        self.MAX_LEN = params['MAX_LEN']
        self.iter = params['iter']
        self.save_path = params['save_path']
        self.pretrain_emb = params['pretrain_emb']
        self.lr = params['lr']
        self.batch_size = params['batch_size']
        self.model = None

    def _build(self):
        '''
         Build your comp graph here, architecture may be changed:
        :param y_unique: will be defined based on your task
        :return:
        '''

        model = Models.Sequential()
        model.add(Embedding(self.NB_WORDS, self.embed_dim, input_length=self.MAX_LEN,
                            mask_zero=True))  # output (None, ma_len, embedding_size)
        model.add(Bidirectional(GRU(self.rnn_out, return_sequences=True), merge_mode='concat'))
        model.add(Bidirectional(GRU(self.rnn_out, return_sequences=True), merge_mode='concat'))
        model.add(Bidirectional(GRU(self.rnn_out, return_sequences=True), merge_mode='concat'))
        # model.add(Bidirectional(GRU(self.rnn_out, return_sequences= True), merge_mode='concat'))
        # model.add(Bidirectional(GRU(self.rnn_out, return_sequences= True), merge_mode='concat'))
        model.add(Bidirectional(GRU(self.rnn_out), merge_mode='concat'))

        model.add(Dense(256,
                        activation='tanh',
                        kernel_initializer='glorot_uniform', # also known as xavier
                        activity_regularizer=regularizers.l2(0.2)))

        model.add(Dense(128,
                        activation='tanh',
                        kernel_initializer='glorot_uniform',
                        activity_regularizer=regularizers.l2(0.2)))

        model.add(Dense(64,
                        activation='tanh',
                        kernel_initializer='glorot_uniform',
                        activity_regularizer=regularizers.l2(.2)))

        model.add(Dense(32,
                        activation='tanh',
                        kernel_initializer='glorot_uniform',
                        activity_regularizer=regularizers.l2(0.2)))

        model.add(Dense(self.y_unique,
                        activation='softmax',
                        kernel_initializer='glorot_uniform',
                        activity_regularizer=regularizers.l2(0.2)))

        if self.pretrain_emb is not None:
            model.layers[0].set_weights([self.pretrain_emb])
            model.layers[0].trainable = False

        self.model = model


    def _compile(self, lr = self.lr):
        '''
        compile the model
        :return: compiled model
        '''

        opt = Adam(lr = lr, epsilon=1e-8, amsgrad=True, clipvalue=0.5)
        if self.y_unique == 2:
            # binary classification
            self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        else:
            # multi classification
            self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        print(self.model.summary())

    ''' get_3rd_layer_output = K.function([model.layers[0].input],
                                          [model.layers[3].output])
        layer_output = get_3rd_layer_output([X])[0]

        self.bottle_neck = K.function([self.model.layers[0].input], [self.model.layers[-2].output])'''


    # training model process
    def train(self, X, y, reset=True):
        # initialize
        val_acc = 0
        batch_size = self.batch_size
        self.y_unique = y.shape[1]

        # build model
        self._build()
        self._compile()
        checkpoint1 = ModelCheckpoint(self.save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5)
        callbacks_list = [checkpoint1, earlystopping]

        self.model.fit(X, \
                       y, \
                       validation_split=.1, \
                       epochs=self.iter, \
                       batch_size=batch_size, \
                       callbacks=callbacks_list, \
                       verbose=1)

    def fine_tune(self, X, y, lr, batch_size = 32):
        '''
        If the performance of training is not improving,
        try reduce learning rate by 1/10 and fine tune the model
        :param X:
        :param y:
        :param lr:
        :return:
        '''
        self._compile(lr = lr)
        # auto save the model with max val_acc
        checkpoint = ModelCheckpoint(self.save_path, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        # stop when the performance hasn't been improved for 3 consequent iterations
        earlystopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        callbacks_list = [checkpoint, earlystopping]

        self.model.fit(X_balanced, \
                       y_balanced, \
                       validation_split=.1, \
                       epochs = 1000, \
                       batch_size= batch_size, \
                       callbacks= callbacks_list, \
                       verbose=1)

    def predict(self, X):
        # if not trained, load best model in history
        if not self.model:
            self._build(2)
        if isfile(self.save_path):
            self.model.load_weights(self.save_path)
        else:
            print('Model not trained.')

        return self.model.predict(X)
