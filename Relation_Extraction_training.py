import os
import sys
import nltk
import spacy
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nlp = spacy.load('en_core_web_sm')
from nltk.corpus import stopwords
import codecs
import gensim
from  gensim import corpora
from collections import defaultdict
# from nltk import tree2conlltags
# import numpy as np
# import pandas as pd
# from gensim import model
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# nltk.download('stopwords')
# nltk.download('treebank')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# POS tag list:
#
# CC	coordinating conjunction
# CD	cardinal digit
# DT	determiner
# EX	existential there (like: "there is" ... think of it like "there exists")
# FW	foreign word
# IN	preposition/subordinating conjunction
# JJ	adjective	'big'
# JJR	adjective, comparative	'bigger'
# JJS	adjective, superlative	'biggest'
# LS	list marker	1)
# MD	modal	could, will
# NN	noun, singular 'desk'
# NNS	noun plural	'desks'
# NNP	proper noun, singular	'Harrison'
# NNPS	proper noun, plural	'Americans'
# PDT	predeterminer	'all the kids'
# POS	possessive ending	parent\'s
# PRP	personal pronoun	I, he, she
# PRP$	possessive pronoun	my, his, hers
# RB	adverb	very, silently,
# RBR	adverb, comparative	better
# RBS	adverb, superlative	best
# RP	particle	give up
# TO	to	go 'to' the store.
# UH	interjection	errrrrrrrm
# VB	verb, base form	take
# VBD	verb, past tense	took
# VBG	verb, gerund/present participle	taking
# VBN	verb, past participle	taken
# VBP	verb, sing. present, non-3d	take
# VBZ	verb, 3rd person sing. present	takes
# WDT	wh-determiner	which
# WP	wh-pronoun	who, what
# WP$	possessive wh-pronoun	whose
# WRB	wh-abverb	where, when



def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def tokenization(corpus):
    token_dict = {}
    filtered_corpus = []
    stop_words = set(stopwords.words('english'))
    sentences = nltk.sent_tokenize(corpus)
    for sentence in sentences:
        word_token = nltk.word_tokenize(sentence)
        filtered_sentece = []
        for w in word_token:
            if w not in stop_words:
                filtered_sentece.append(w)
        filtered_corpus.append(filtered_sentece)
        token_dict[sentence] = word_token
    return token_dict, filtered_corpus

#POS Tagging
def pos_tagging(token_dict):
    pos_tag_dict = {}
    for key in token_dict:
        token_list = token_dict[key]
        pos_tagged = nltk.pos_tag(token_list)
        pos_tag_dict[key] = pos_tagged
    return pos_tag_dict

#Lemmatize
def lemmatization(pos_tag_dict):
    lemma_dict = {}
    lemmatizer = WordNetLemmatizer()
    for key in pos_tag_dict:
        pos_tag_list = pos_tag_dict[key]
        for  word_pos in pos_tag_list:
            lemma_dict[word_pos[0]+"_"+word_pos[1]] = lemmatizer.lemmatize(word_pos[0],pos=get_wordnet_pos(word_pos[1]))
    return lemma_dict

def chunking(pos_tag_dict):
    grammar = r"""
        NP: { < DT | JJ | NN. * > +}  # Chunk sequences of DT, JJ, NN
        PP: { < IN > < NP >}  # Chunk prepositions followed by NP
        VP: { < VB. * > < NP | PP | CLAUSE > +$}  # Chunk verbs and their arguments
      """
    parse_dict = {}
    cp = nltk.RegexpParser(grammar)
    for sentence, pos_tag_sentence in pos_tag_dict.items():
        parse_dict[sentence] = cp.parse(pos_tag_sentence)
        # parse_dict[sentence] = tree2conlltags(non_iob)
    return parse_dict

def named_entity_recognition(parse_dict):
    ner_dict = defaultdict(list)
    for sentence, parse_tree in parse_dict.items():
      # ner_dict[sentence] = nltk.ne_chunk(parse_tree)
        doc = nlp(sentence)
        for tuple in doc.ents:
            ner_dict[sentence].append([tuple.text, tuple.label_])
    return ner_dict

def stemming(token_dict):
    stemmer = nltk.stem.PorterStemmer()
    stem_dict = {}
    for sentence in token_dict.keys():
         stems = " "
         token_list = token_dict[sentence]
         for token in token_list:
             stem =  " "+stemmer.stem(token)
             stems += stem
         stem_dict[sentence] = stems
    return stem_dict

def create_bag_of_words(filtered_corpus):
    bag_of_words = {}
    for sentence in filtered_corpus:
        for word in sentence:
            if word in bag_of_words.keys():
                bag_of_words[word] += 1
            else:
                bag_of_words[word] = 1
    return bag_of_words


if __name__ == "__main__":
    training_data_folder = sys.argv[1];
    files = []
    for entry in os.listdir(training_data_folder):
        if os.path.isfile(os.path.join(training_data_folder, entry)):
            files.append(os.path.join(training_data_folder, entry))
    corpus_raw = u""
    for file in files:
        with codecs.open(file, "r" , "utf-8") as text:
            corpus_raw += text.read()

    token_dict, filtered_corpus = tokenization(corpus_raw)
    pos_tag_dict = pos_tagging(token_dict)
    lemma_dict = lemmatization(pos_tag_dict)
    stem_dict = stemming(token_dict)
    parse_dict = chunking(pos_tag_dict)
    bag_of_words = create_bag_of_words(filtered_corpus)
    ner_dict = named_entity_recognition(parse_dict)
    # dictionary = gensim.corpora.Dictionary(filtered_corpus)
    # dictionary.filter_extremes(no_below=5,no_above=0.8)
    # bag_of_words = [dictionary.doc2bow(sentence) for sentence in filtered_corpus]
    # tfidf = models.TfidfModel(bag_of_words)
    # corpus_tfidf = tfidf[bag_of_words]
    # tfidf = TfidfVectorizer(min_df=2,max_df=0.5, ngram_range=(1,2))
    # features = tfidf.fit_transform(filtered_corpus)
    # pd.DataFrame(features.todense(),columns=tfidf.get_feature_names())
    # lda_model = gensim.models.LdaMulticore(bag_of_words,num_topics=3,id2word=dictionary,passes=2)
    # for ids,topic in lda_model.print_topics(-1):
    #     print("Topic: {} words: {}".format(ids,topic) )
    #     print("\n")



