# file loading and os packages
import os
from urllib.request import urlretrieve
from os.path import isfile, isdir, join
import shutil
from sklearn.metrics import f1_score,accuracy_score

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
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import backend as K



# local embedded model params
'''
path = 'path/to/your/file'
pretrain_path = join(path, 'glove.twitter.27B')
'''



def get_embeddings(path_to_emb):
    '''
    :param path_to_emb: defined above
    :return:
    '''
    emb_dict = {}
    with open(path_to_emb, 'r') as glv:
        for line in glv:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            emb_dict[word] = vector
        glv.close()
    return emb_dict

def get_emb_matrix(GLOVE_DIM, emb_dict, tk, NB_WORDS = 10000):
    '''
    :param NB_WORDS
    :param GLOVE_DIM: your glove embedding dimension
    :param emb_dict: the dictionary of glove
    :param tk: tokenizer you generate from TextPipeline
    :return:
    '''
    # collect glove word embeddings:
    emb_matrix = np.zeros((NB_WORDS, GLOVE_DIM))
    for w, i in tk.word_index.items():
        if i == 0:
            continue
        if i < NB_WORDS:
            vector = emb_dict.get(w)
            if vector is not None:
                emb_matrix[i] = vector
        else:
            break
    return emb_matrix

'''

Load your glove dictionary and transform to embedding matrix
# load glove 50d 
glv50_dict = get_embeddings(join(pretrain_path, 'glove.twitter.27B.50d.txt'))
glv100_dict = get_embeddings(join(pretrain_path, 'glove.twitter.27B.100d.txt'))
glv50_matrix = get_emb_matrix(50, glv50_dict, tk)
glv100_matrix = get_emb_matrix(100, glv100_dict, tk)

'''