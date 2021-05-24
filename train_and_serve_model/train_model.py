import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import json
from bs4 import BeautifulSoup
import unicodedata
import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import pickle
from scipy import sparse as sp_sparse

import warnings
warnings.filterwarnings('ignore')

np.random.seed(10)

# Below are the functions that will contribute to the pre-processing of the dataset.
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def special_characters_removal(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


def create_bow(text, word_ind, dict_size):
    res_vec = np.zeros(dict_size)
    for rec in text.split(' '):
        if rec in word_ind:
            res_vec[word_ind[rec]] += 1
    return res_vec


# Dataset is in json format therefore I shall be using the below method to load the data into the data structure before
# going further
with open('sampleJobDataWithTags.json', encoding="utf-8") as dataset_json:
    training_dataset = json.load(dataset_json)

preprocessed_dataset = []
lemmatizer = WordNetLemmatizer()

# Data cleansing
for datarecords in training_dataset:

    # The BeautifulSoap library helps to scrape data from webpages and provides with the html parser to get the text from
    # html which is exactly what is being done below.
    datarecords["title"] = BeautifulSoup(
        datarecords["title"], "html.parser").get_text()
    datarecords["title"] = unicodedata.normalize('NFKD', datarecords["title"]).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    datarecords["title"] = special_characters_removal(datarecords["title"])
    datarecords["title"] = datarecords["title"].lower()

    datarecords["description"] = BeautifulSoup(
        datarecords["description"], "html.parser").get_text()
    datarecords["description"] = unicodedata.normalize(
        'NFKD', datarecords["description"]).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    datarecords["description"] = special_characters_removal(
        datarecords["description"])
    datarecords["description"] = datarecords["description"].lower()

    # Performing lemmatization
    # First tokenizing and splitting the data into words before performing lemmatization
    wordlist_title = nltk.word_tokenize(datarecords["title"])
    wordlist_description = nltk.word_tokenize(datarecords["description"])
    datarecords["title"] = ' '.join(
        [lemmatizer.lemmatize(lemw) for lemw in wordlist_title])
    datarecords["description"] = ' '.join(
        [lemmatizer.lemmatize(lemw) for lemw in wordlist_description])

    # appending the data
    preprocessed_dataset.append(
        {"title": datarecords["title"], "description": datarecords["description"], "tags": datarecords["tags"]})

print("Preprocessed Dataset: \n")
print(preprocessed_dataset[0:1])

# Converting json array to dataframe
trainset = pd.read_json(json.dumps(preprocessed_dataset))

# Removing of the Stop Words
stop_words = set(stopwords.words('english'))
trainset['description'] = trainset["description"].apply(
    lambda t: remove_stopwords(t))

multilb = MultiLabelBinarizer()
multilb.fit(trainset['tags'])

# transform target variable
target = multilb.transform(trainset['tags'])

trainX, testX, ytrain, yval = train_test_split(
    trainset['description'], target, test_size=0.20, random_state=9)

count_of_words = {}
for desc in trainX:
    for token in desc.split():
        if token not in count_of_words:
            count_of_words[token] = 1
        count_of_words[token] += 1

size_of_dict = 10000
pop_words = sorted(count_of_words, key=count_of_words.get,
                   reverse=True)[:size_of_dict]
words_index = {key: rank for rank, key in enumerate(pop_words, 0)}
index_words = {index: word for word, index in words_index.items()}
everyword = words_index.keys()

trainX_bow = sp_sparse.vstack([sp_sparse.csr_matrix(
    create_bow(descr, words_index, size_of_dict)) for descr in trainX])
testX_bow = sp_sparse.vstack([sp_sparse.csr_matrix(
    create_bow(descr, words_index, size_of_dict)) for descr in testX])
print('X_train shape ', trainX_bow.shape, '\nX_val shape ', testX_bow.shape)


lr = LogisticRegression()
clf = OneVsRestClassifier(lr)

# fitting the model on training data
clf.fit(trainX_bow, ytrain)

filename = 'model.pkl'
pickle.dump(clf, open(filename, 'wb'))

filename = 'multilb.pkl'
pickle.dump(multilb, open(filename, 'wb'))

filename = 'wordsindex.pkl'
pickle.dump(words_index, open(filename, 'wb'))

filename = 'preprocessed_trainset.pkl'
pickle.dump(trainset, open(filename, 'wb'))
