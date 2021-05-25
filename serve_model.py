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

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

loaded_model = pickle.load(open("model.pkl", "rb"))  # read binary
loaded_wordsindex = pickle.load(open("wordsindex.pkl", "rb"))  # read binary
loaded_multilb = pickle.load(open("multilb.pkl", "rb"))  # read binary
loaded_preprocessed_trainset = pickle.load(
    open("preprocessed_trainset.pkl", "rb"))

with open('sampleJobDataWithTags.json', encoding="utf-8") as dataset_json:
    training_dataset = json.load(dataset_json)
    print("Total Training samples", len(training_dataset))


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


def MakeDecision(p_value):
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

    return all_tags_dfa


def monitor_model_out(target_predictions_history):
    td_dist = TrainingData_Tags_Distribution()
    tp_hist = frequency_of_words(target_predictions_history)

    # Welsh's t-test assumes unequal sample sizes
    decision = MakeDecision(stats.ttest_ind(
        td_dist['count'], tp_hist['count'], equal_var=False).pvalue)

    return decision


def monitor_model_inp(input_descriptions_history):
    input_train_dist = frequency_of_words(
        loaded_preprocessed_trainset["description"])
    input_descs_dist = frequency_of_words(input_descriptions_history)

    # Welshs t-test assumes unequal sample sizes
    decision = MakeDecision(stats.ttest_ind(
        input_train_dist['count'], input_descs_dist['count'], equal_var=False).pvalue)

    return decision


@app.route('/monitorinput', methods=['POST'])
def monitor_model_input():
    if request.method == 'POST':
        print("1")
        data = request.get_json(force=True)
        input_descriptions_history = data['desc']

        input_train_dist = frequency_of_words(
            loaded_preprocessed_trainset["description"])
        input_descs_dist = frequency_of_words(input_descriptions_history)

        # Welshs t-test assumes unequal sample sizes
        decision = MakeDecision(stats.ttest_ind(
            input_train_dist['count'], input_descs_dist['count'], equal_var=False).pvalue)

    return decision


@app.route('/monitoroutput', methods=['POST'])
def monitor_model_output():
    # finding tags distributions
    if request.method == 'POST':
        print("2")
        data = request.get_json(force=True)
        target_predictions_history = data['desc']

        td_dist = TrainingData_Tags_Distribution()
        tp_hist = frequency_of_words(target_predictions_history)

        # Welsh's t-test assumes unequal sample sizes
        decision = MakeDecision(stats.ttest_ind(
            td_dist['count'], tp_hist['count'], equal_var=False).pvalue)

    return decision


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predictresult', methods=['POST'])
def predictionresult():
    if request.method == 'POST':
        data = request.get_json(force=True)
        description = str(data['desc'])

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

        if len(prediction[0]) != 0:
            print(str(prediction[0][0]))
            f = open("descriptions.txt", 'a')
            f.write(description)
            f.write('\n')

            predictions_list = ""
            f = open("predictions.txt", 'a')
            for i in range(len(prediction[0])):
                f.write(prediction[0][i])
                predictions_list += str(prediction[0][i]) + "\n"
                f.write('\n')

            return predictions_list

    return "No Tags Inferred"


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
        print(len(prediction))

        if len(prediction[0]) != 0:
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

            input_decision = monitor_model_inp(history_descriptions)
            print(input_decision)
            output_decision = monitor_model_out(history_predictions)
            print(output_decision)

            return render_template("prediction.html", prediction=str(prediction))

    return "No Tags inferred"


if __name__ == "__main__":
    # docker build -t nlpproject:latest .
    # docker run -it -p 5000:5000 nlpproject

    app.run(host='0.0.0.0', port=8080, debug=True)
