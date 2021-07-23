import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.utils import shuffle
import pickle
import spacy
import unicodedata
import re
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from collections import defaultdict
from scipy.stats import friedmanchisquare
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from nltk.corpus import stopwords
import time
from sklearn.metrics import confusion_matrix
import cv2
import os
from tqdm import tqdm
import math
import nltk
import string
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

nlp_spacy = spacy.load('en_core_web_sm')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import os

def read(address):
    arq = open(address, 'rb')
    return pickle.load(arq)


def save_pickle(address, element):
    arq = open(address, 'wb')
    pickle.dump(element, arq)


def save_data_frame(s, name):
    s.to_csv(name + '.csv', index=False)

### MUDAR FONTE
def return_domain(domain_name):
    return read("drive/My Drive/sa/DATASET/"+domain_name+".pk")

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

neg_words = ['not', 'no', "n't", "never", "nothing"]
def tokenize(doc):
    stop = stopwords.words('english')
    x = []
    t = word_tokenize(doc)
    for a in t:
        a = a.lower()
        if a in "n't":
                x.append("not")
        elif a not in stop or a in neg_words:
            if a.isalpha() and len(a) > 1:
                x.append(WordNetLemmatizer().lemmatize(a))
    return x
  

# The input is a tokenize document
pos_import = ["NN", "VB", "JJ", "RB"]


def pos_filter(token_doc,nword):
    pos = nltk.pos_tag(token_doc)
    new_t = []
    jump = -1
    for p in pos:
        k = [True if po in p[1] else False for po in pos_import]
        if any(k) or ((p[0] in neg_words)):
            if p[0] in neg_words and nword:
                n, jump = neg_affect(p, pos, 4)
                if n != "":
                    new_t.append(n)
            else:
                if pos.index(p) == jump:
                    pass
                else:
                    new_t.append(p[0])

    return new_t
  
# w is window
def neg_affect(word, pos, w):
    neg_pos = ["VB", "JJ", "RB"]
    i = pos.index(word)
    if word[0] == "no":
        try:
            if "NN" in pos[i + 1][1]:
                n = pos[i + 1][0] + "_NOT"
                return n, i + 1
        except:
            return "", -1
    else:
        for j in range(i + 1, i + w):
            try:
                k = [True if po in pos[j][1] else False for po in neg_pos]
                if any(k):
                    n = pos[j][0] + "_NOT"
                    return n, j
                elif pos[j][1] in punct:
                    return "", -1
            except:
                return "", -1
    return "", -1

def n_gr(token_doc):
    new_token = []
    for i in range(len(token_doc)):
        if i < (len(token_doc) - 1):
            new_token.append(token_doc[i])
            bi_gram = "%s__%s" % (token_doc[i], token_doc[i + 1])
            new_token.append(bi_gram)
    return new_token

def pipeline(doc):
    t = tokenize(doc)
    nt = pos_filter(t,True)
    nt_gram = n_gram(nt)
    return nt_gram


def thr_lim(all_doc,thr):
  dic = word_frequency(all_doc)
  new_rev = []
  for i in list(dic):
    if dic[i] < thr:
      del(dic[i])
  for doc in all_doc:
    new = []
    for w in doc:
      try:
        dic[w]
        new.append(w)
      except:
        pass
    new_rev.append(new)
    
  return new_rev,dic


def word_frequency(all_doc):
  dic = {}
  vocab = []
 
  for token in all_doc:
    for w in token:
      if w in vocab:
        dic[w] += 1
      else:
        vocab.append(w)
        dic[w] = 1
  return dic


def preprocess(token_rev, nword = True, pos = True, ng = True, thr = 0 , wrdfreq = False):
  prepData = []
  
  for doc in (token_rev):
    prepData.append(selecPipe(doc, nword, pos, ng))
  if thr > 0:
    prepData, wordFreq = thr_lim(prepData,thr)
    return prepData, wordFreq
  if thr < 1 and wrdfreq:
    wordFreq = word_frequency(prepData)
    return prepData, wordFreq

  return prepData




def selecPipe(doc, nword, pos, ng):
  t = tokenize(doc)
  if pos == True:
    t = pos_filter(t,nword) 
  if ng == True:
    t = n_gr(t)
  return t 


def generateembedding_matrix(texts,maxlen = 0, ng = False, nword = False ,thr = 1):


    preprocessed = preprocess(texts,nword=nword, ng = ng)
    model = Word2Vec(preprocessed, min_count=thr)

    embed_size = 100

    t = Tokenizer()
    t.fit_on_texts(preprocessed)

    if maxlen == 0:
    	maxlen = len(max(preprocessed,key=len))

    vocab_size = len(t.word_index) + 1

    

    embedding_matrix = np.zeros((vocab_size, 100))

    for word, i in t.word_index.items():
        embedding_vector = model.wv[word]
        if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector


    return preprocessed,t, embedding_matrix,maxlen,vocab_size,embed_size


def pre_processing_keep_stopword_token(texts): 

    textsretorno = []

    for text in texts:
      text = nlp_spacy(text) 
      textsretorno.append([word.lower_ for word in text if word.is_digit==False and word.pos_!="SYM" and word.is_punct==False])
    
    return textsretorno


def generate_cnndata(texts, path = None, maxlen = 0, preprocesstype = 'new',pad = 'pre',size = 100,ng = False, nword = False ,thr = 1):


  if preprocesstype == 'new':
      preprocessed = pre_processing_keep_stopword_token(texts)

  if preprocesstype == 'old':
      preprocessed = preprocess(texts,nword=nword, ng = ng)




  if path == None:
    model = Word2Vec(preprocessed,size = size, min_count=thr)
  else:
    model_2 = Word2Vec(size=size, min_count=thr)
    model_2.build_vocab(preprocessed)
    total_examples = model_2.corpus_count
    model = KeyedVectors.load_word2vec_format(path, binary=False)
    model_2.build_vocab([list(model.vocab.keys())], update=True)
    model_2.intersect_word2vec_format(path, binary=False, lockf=1.0)
    model = model_2.train(preprocessed, total_examples=total_examples, epochs=model_2.iter)


  t = Tokenizer()
  t.fit_on_texts(preprocessed)

  if maxlen == 0:
      maxlen = len(max(preprocessed,key=len))

  data = apply_transformpad(preprocessed, t, maxlen,pad)

  
  newdata = np.zeros((len(data),maxlen, size))

  for i,sequen in enumerate(data):
    for index,ele in enumerate(sequen):
      if ele == 0:
          continue
      else:              
          for word, value in t.word_index.items():
              if value == ele:
                try:
                  newvector = model.wv[word]
                except:
                  newvector = np.zeros(100)
                newdata[i][index] = newvector
                break
              else:
                  continue

  return newdata,data,preprocessed, newdata.shape[1],newdata.shape[2], model


def apply_transformpad(sequences, t, maxlen, pad = 'pre'):

    encoded_docs = t.texts_to_sequences(sequences)
    return sequence.pad_sequences(encoded_docs, maxlen=maxlen,padding = pad)