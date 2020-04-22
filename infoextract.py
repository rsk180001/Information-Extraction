import sys
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer



def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens


def lemmatizing(tokenized_text):
    wn = nltk.WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text


def clean_text(text):
    tokens = re.split('\W+', text)
    wn = nltk.WordNetLemmatizer()
    text2 = [wn.lemmatize(word) for word in tokens]
    return text2


if len(sys.argv) > 1:
    input = sys.argv[1]
    pd.set_option('display.max_colwidth', 100)
    data = pd.read_csv(input, sep='\t', header=None)
    data.columns = ['body_text']
    data_sample = data[0:20]
    tfidf_vect_sample = TfidfVectorizer(analyzer=clean_text)
    X_tfidf_sample = tfidf_vect_sample.fit_transform(data_sample['body_text'])
    X_tfidf_df = pd.DataFrame(X_tfidf_sample.toarray())
    X_tfidf_df.columns = tfidf_vect_sample.get_feature_names()
    X_features = pd.DataFrame(X_tfidf_df)
    print(X_features)
else:
    print("Please provide input file!!!")