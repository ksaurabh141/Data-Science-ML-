# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:22:34 2018

@author: Ankita
"""
import numpy as np
import pandas as pd
from keras.layers import Dense,Dropout,Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.preprocessing.text import Tokenizer

from nltk import word_tokenize
from nltk.corpus import stopwords
import re

train_dir = pd.read_csv('C:\\Data Science\\Deep Learning\\data\\NLP\\Spooky Author Identification\\train\\train.csv')
test_dir = pd.read_csv('C:\\Data Science\\Deep Learning\\data\\NLP\\Spooky Author Identification\\test\\test.csv')

glove_file = 'C:\\Data Science\\Deep Learning\\data\\NLP\\glove.6B\\glove.6B.50d.txt'
word_embed_size = 50
batch_size = 70
epochs = 30
sentence = train_dir['text']

def cleanSentence(sentence):
    print("Actual Sentence: " + sentence)
    sentence_clean= re.sub("[^a-zA-Z]"," ", sentence)
    print(("/t"))
    print("Clean sentence: " + sentence_clean)
    sentence_clean = sentence_clean.lower()
    tokens = word_tokenize(sentence_clean)
    print(("/t"))
    print("Tokens: ")
    print(tokens)
    stop_words = set(stopwords.words("english"))
    print(("/t"))
    print("Stop Words: ")
    print(stop_words)
    sentence_clean_words = [w for w in tokens if not w in stop_words]
    print(("\t"))
    print("Sentence Clean Words: ")
    print(sentence_clean_words)
    print(("\t"))    
    print(("_________________________"))  
    return ' '.join(sentence_clean_words)

sentence = list(map(cleanSentence, sentence))

def BuildVocabulary(sentence):
    text = list(sentence)
    print('text: ')
    print(text)
    print(('\t'))
    tokenizer = Tokenizer(lower= False, split= ' ')
    print('tokenizer: ')
    print(tokenizer)
    print(('\t'))
    tokenizer.fit_on_texts(text)
    print(tokenizer)
    print(('\t'))
    return tokenizer

tokenizer = BuildVocabulary(sentence)
vocab_size = len(tokenizer.word_index)+1
print(vocab_size)

def get_train_sequence(sentence,tokenizer):
    sent = tokenizer.texts_to_sequences(sentence)
    sent_maxlen = max([len(s) for s in sent])
    seq_maxlen = max([sent_maxlen])
    return np.array(pad_sequences(sent, maxlen= seq_maxlen))
    
x_train = get_train_sequence(sentence, tokenizer)
y_train = train_dir['author']
seq_maxlen = len(x_train[0])
print(seq_maxlen)

def loadGloveWordEmbeddings(glove_file):
    embedding_vectors = {}
    f = open(glove_file, encoding = 'UTF-8')
    for line in f:
        values = line.split()
        word = values[0]
        value = np.asarray(values[1:], dtype= 'float32')
        embedding_vectors[word] = value
    f.close()
    return embedding_vectors

embedding_vectors = loadGloveWordEmbeddings(glove_file)
print(len(embedding_vectors))

def getEmbeddingWeightMatrix(embedding_vectors, word2idx): 
    #word_embed_size is Glove file dimension size
    embedding_matrix = np.zeros((len(word2idx)+1, word_embed_size))
    for word, i in word2idx.items():
        embedding_vector = embedding_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

embedding_weight_matrix = getEmbeddingWeightMatrix(embedding_vectors, tokenizer.word_index)
print(embedding_weight_matrix.shape)

#Now build a Embedding layer
train_Input = Input(shape=(x_train.shape[1],),dtype = 'int32')
#Build an Embedding layer
left = Embedding(input_dim = vocab_size, output_dim= word_embed_size, input_length= seq_maxlen, weights = [embedding_weight_matrix],trainable = False) (train_Input)
left = LSTM(100, return_sequences=False)(left)
left = Dropout(0.3)(left)

x = Dense(10, activation='relu')(left)
output = Dense(3)(x) #Tells the similarity

model = Model(inputs =train_Input , outputs = output)
print(model.summary())


