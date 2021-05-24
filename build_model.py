from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import seaborn as sns
import matplotlib.pyplot as plt
import re
import unicodedata
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
from scipy import sparse as sp_sparse
import scipy.stats as stats

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import warnings
warnings.filterwarnings('ignore')

np.random.seed(10)

def create_bow(text, word_ind, dict_size):
    res_vec = np.zeros(dict_size)
    for rec in text.split(' '):
        if rec in word_ind:
            res_vec[word_ind[rec]] += 1
    return res_vec


def special_characters_removal(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


def frequency_of_words(x):
    all_words = ''.join([text for text in x])
    all_words = all_words.split()
    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame(
        {'word': list(fdist.keys()), 'count': list(fdist.values())})
    # selecting top n most frequent words
    #d = words_df.nlargest(columns="count", n=terms)
    words_df = words_df.sort_values("count")
    print(words_df.head())

    return words_df


def ComputeChiSquareGOF(expected, observed):
    """
    Runs a chi-square goodness-of-fit test and returns the p-value.
    Inputs:
    - expected: numpy array of expected values.
    - observed: numpy array of observed values.
    Returns: p-value
    """
    expected_scaled = expected / float(sum(expected)) * sum(observed)
    result = stats.chisquare(f_obs=observed, f_exp=expected_scaled)
    return result[1]


def MakeDecision(p_value):
    """ 
    Makes a goodness-of-fit decision on an input p-value.
    Input: p_value: the p-value from a goodness-of-fit test.
    Returns: "different" if the p-value is below 0.05, "same" otherwise
    """
    return "different" if p_value < 0.05 else "same"


def TrainingData_Tags_Distribution():
    distribution = {}
    for datarec in training_dataset:
        for tagg in datarec["tags"]:
            if tagg in distribution.keys():
                distribution[tagg] += 1
            else:
                distribution[tagg] = 1

    all_tags_dfa = pd.DataFrame({'Tag': list(distribution.keys()),
                                'count': list(distribution.values())})

    all_tags_dfa = all_tags_dfa.sort_values("count")
    #print("\n\nTraining data tags distribution for the first 100 tags: ")
    #ga = all_tags_dfa.nlargest(columns="Count", n=100)
    return all_tags_dfa


def monitor_model_input(input_descriptions_history):
    input_train_dist = frequency_of_words(
        loaded_preprocessed_trainset["description"])
    input_descs_dist = frequency_of_words(input_descriptions_history)

    # p_value = ComputeChiSquareGOF(
    #   input_train_dist['count'], input_descs_dist['count'])

    # Welshs t-test assumes unequal sample sizes
    decision = MakeDecision(stats.ttest_ind(
        input_train_dist['count'], input_descs_dist['count'], equal_var=False).pvalue)

    return decision


def monitor_model_output(target_predictions_history):
    # finding tags distributions
    td_dist = TrainingData_Tags_Distribution()
    tp_hist = frequency_of_words(target_predictions_history)

    # Welsh's t-test assumes unequal sample sizes
    decision = MakeDecision(stats.ttest_ind(
        td_dist['count'], tp_hist['count'], equal_var=False).pvalue)

    return decision



#Train and Load the model

app = Flask(__name__)

#Dataset is in json format therefore I shall be using the below method to load the data into the data structure before 
#going further
with open('sampleJobDataWithTags.json', encoding="utf-8") as dataset_json:
	training_dataset = json.load(dataset_json)

preprocessed_dataset = []
lemmatizer = WordNetLemmatizer()

# Data cleansing
for datarecords in training_dataset:
    
    # The BeautifulSoap library helps to scrape data from webpages and provides with the html parser to get the text from 
    # html which is exactly what is being done below. 
	datarecords["title"] = BeautifulSoup(datarecords["title"], "html.parser").get_text()
	datarecords["title"] = unicodedata.normalize('NFKD', datarecords["title"]).encode('ascii', 'ignore').decode('utf-8', 'ignore')
	datarecords["title"]  = special_characters_removal(datarecords["title"])
	datarecords["title"] = datarecords["title"].lower()
    
	datarecords["description"] = BeautifulSoup(datarecords["description"], "html.parser").get_text() 
	datarecords["description"] = unicodedata.normalize('NFKD', datarecords["description"]).encode('ascii', 'ignore').decode('utf-8', 'ignore')    
	datarecords["description"]  = special_characters_removal(datarecords["description"])
	datarecords["description"] = datarecords["description"].lower()
    
    # Performing lemmatization
    # First tokenizing and splitting the data into words before performing lemmatization
	wordlist_title = nltk.word_tokenize(datarecords["title"])
	wordlist_description = nltk.word_tokenize(datarecords["description"])
	datarecords["title"] = ' '.join([lemmatizer.lemmatize(lemw) for lemw in wordlist_title])
	datarecords["description"] = ' '.join([lemmatizer.lemmatize(lemw) for lemw in wordlist_description])

	# appending the data
	preprocessed_dataset.append({"title": datarecords["title"], "description": datarecords["description"], "tags": datarecords["tags"]})

print("Preprocessed Dataset: \n")
print(preprocessed_dataset[0:1])

# Converting json array to dataframe
trainset = pd.read_json(json.dumps(preprocessed_dataset))

#Removing of the Stop Words
stop_words = set(stopwords.words('english'))
trainset['description'] = trainset["description"].apply(lambda t: remove_stopwords(t))

multilb = MultiLabelBinarizer()
multilb.fit(trainset['tags'])

# transform target variable
target = multilb.transform(trainset['tags'])

trainX, testX, ytrain, yval = train_test_split(trainset['description'], target, test_size=0.20, random_state=9)

count_of_words = {}
for desc in trainX:
    for token in desc.split():
        if token not in count_of_words:
            count_of_words[token] = 1
        count_of_words[token] += 1

size_of_dict = 10000
pop_words = sorted(count_of_words, key=count_of_words.get, reverse=True)[:size_of_dict]
words_index = {key: rank for rank, key in enumerate(pop_words, 0)}
index_words = {index:word for word, index in words_index.items()}
everyword = words_index.keys()

trainX_bow = sp_sparse.vstack([sp_sparse.csr_matrix(create_bow(descr, words_index, size_of_dict)) for descr in trainX])
testX_bow = sp_sparse.vstack([sp_sparse.csr_matrix(create_bow(descr, words_index, size_of_dict)) for descr in testX])
print('X_train shape ', trainX_bow.shape, '\nX_val shape ', testX_bow.shape)

from sklearn.linear_model import LogisticRegression 
from sklearn.multiclass import OneVsRestClassifier

lr = LogisticRegression() 
clf = OneVsRestClassifier(lr)

#fitting the model on training data
clf.fit(trainX_bow, ytrain)

filename='model.pkl'
pickle.dump(clf, open(filename, 'wb'))

filename='multilb.pkl'
pickle.dump(multilb, open(filename, 'wb'))

filename='wordsindex.pkl'
pickle.dump(words_index, open(filename, 'wb'))

filename='preprocessed_trainset.pkl'
pickle.dump(trainset, open(filename, 'wb'))

loaded_model = pickle.load(open("model.pkl", "rb"))  # read binary
loaded_wordsindex = pickle.load(open("wordsindex.pkl", "rb"))  # read binary
loaded_multilb = pickle.load(open("multilb.pkl", "rb"))  # read binary
loaded_preprocessed_trainset = pickle.load(
    open("preprocessed_trainset.pkl", "rb"))

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def result():
    if request.method == 'POST':
        # We build products and services powered by payments data to find and stop financial crime. By combining data science technique with an intimate knowledge of payments data we develop solutions that will improve outcomes for people, businesses and economies. Headquartered in The City of London, we craft bespoke algorithms that help our clients gain an understanding of the underlying criminal behaviour that drives financial crime.As a Data Scientist, you will join one of the first teams in the world looking at payments data in the UK and across the world. In the research discipline you will help build systems that expose money laundering and detect fraud, managing other data scientists and working with clients to understand the underlying behaviours employed by criminals. You will be product focused, working in close collaboration with our engineering and operations data scientists as well as the wider sales, consulting, and product teams.
        description = request.form.get('desc')

        # Preprocessing the description input
        description = BeautifulSoup(description, "html.parser").get_text()
        description = unicodedata.normalize('NFKD', description).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        description = special_characters_removal(description)
        description = description.lower()
        description = remove_stopwords(description)

        wordlist_description = nltk.word_tokenize(description)
        lemmatizer = WordNetLemmatizer()
        description = ' '.join([lemmatizer.lemmatize(lemw)
                               for lemw in wordlist_description])

        test_bow = sp_sparse.vstack([sp_sparse.csr_matrix(
            create_bow(description, loaded_wordsindex, 10000))])

        y_pred_prob = loaded_model.predict_proba(test_bow)

        t = 0.3  # setting the threshold value
        result = (y_pred_prob >= t).astype(int)
        prediction = loaded_multilb.inverse_transform(result)
        print(str(prediction[0][0]))

        f = open("descriptions.txt", 'a')
        f.write(description)
        f.write('\n')

        f = open("predictions.txt", 'a')
        for i in range(len(prediction[0])):
            f.write(prediction[0][i])
            f.write('\n')

        f = open("descriptions.txt", 'r')
        history_descriptions = f.read().replace("\n", " ")
        f.close()
        f = open("predictions.txt", 'r')
        history_predictions = f.read().replace("\n", " ")
        f.close()

        input_decision = monitor_model_input(history_descriptions)
        print(input_decision)
        output_decision = monitor_model_output(history_predictions)
        print(output_decision)

    return render_template("prediction.html", prediction=str(prediction))


if __name__ == "__main__":
    # docker build -t nlpproject:latest .
    # docker run -it -p 5000:5000 nlpproject

    # docker build -t --pull -t deploy .
    # docker run -it -p 5000:5000 deploy
    app.run(host='0.0.0.0', port=5000, debug=True)
