import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

from MoreFunctions.code import preprocess
from MoreFunctions.code import models

from MoreFunctions.innvestigate import create_analyzer

import nltk
import spacy
nlp_spacy = spacy.load('en_core_web_sm')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def apply_transformpad(sequences, t, maxlen, pad = 'pre'):

    encoded_docs = t.texts_to_sequences(sequences)
    return sequence.pad_sequences(encoded_docs, maxlen=maxlen,padding = pad)

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

def pre_processing_keep_stopword_token(texts): 

    textsretorno = []

    for text in texts:
      text = nlp_spacy(text) 
      textsretorno.append([word.lower_ for word in text if word.is_digit==False and word.pos_!="SYM" and word.is_punct==False])
    
    return textsretorno

def generate_cnndata(texts, model = None, path = None, maxlen = 0, preprocesstype = 'new',pad = 'pre',size = 100,ng = False, nword = False ,thr = 1):

  # When preprocesstype is equal to new
  # the pre process call the pre_processing_keep_stopword_token function
  # that only clean the numbers and punctuations from the dataset

  if preprocesstype == 'new':
      preprocessed = pre_processing_keep_stopword_token(texts)

  if preprocesstype == 'old':
      preprocessed = preprocess(texts,nword=nword, ng = ng)


  if path == None:
    model = Word2Vec(preprocessed,size = size, min_count=thr)
  else:

    # Here is the part where i add the embeddings of the glove to a
    # word2vec model then i train the model with the words that is out
    # of the glove vocabulary

    # that function should work fine with your dataset too

    model_2 = Word2Vec(size=size, min_count=thr)
    model_2.build_vocab(preprocessed)
    total_examples = model_2.corpus_count
    model_2.build_vocab([list(model.vocab.keys())], update=True)
    model_2.intersect_word2vec_format(path, binary=False, lockf=1.0)
    model_2.train(preprocessed, total_examples=total_examples, epochs=model_2.iter)


  t = Tokenizer()
  t.fit_on_texts(preprocessed)

  if maxlen == 0:
      maxlen = len(max(preprocessed,key=len))

  # Here i pad all the sentences to the same length
  data = apply_transformpad(preprocessed, t, maxlen,pad)

  newdata = np.zeros((len(data),maxlen, size))

  # In this part, for every sentence i get the word embedding representation
  # from the model trained using glove and word2vec and i put in the 
  # newdata correct position, in the end i get an array of matrix containing 
  # every sentence of the dataset and the word embedding for each of word of the sentence
  # Basically, every sentence is a matrix of the same height and width.
  # I dont know if i explained well, but if you did not understand, you can say to me and i explain again.

  for i,sequen in enumerate(data):
    for index,ele in enumerate(sequen):
      if ele == 0:
          continue
      else:              
          for word, value in t.word_index.items():
              if value == ele:
                try:
                  newvector = model_2.wv[word]
                except:
                  newvector = np.zeros(100)
                newdata[i][index] = newvector
                break
              else:
                  continue

  return newdata,data,preprocessed, newdata.shape[1],newdata.shape[2], model_2