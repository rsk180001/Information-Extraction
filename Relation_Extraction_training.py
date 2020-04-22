import os
import sys
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
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

#Tokenization
def tokenization(files):
    token_dict = {}
    for file in files:
        f = open(file, "r", encoding='utf-8')
        if f.mode == 'r':
            text = f.read()
            token_text = sent_tokenize(text)
            for sentence in token_text:
                word_token = nltk.word_tokenize(sentence)
                token_dict[sentence] = word_token
    return token_dict

#POS Tagging
def pos_tagging(token_dict):
    pos_tag_dict = {}
    for key in token_dict:
        token_list = token_dict[key]
        pos_tagged = nltk.pos_tag(token_list)
        pos_tag_dict[key] = pos_tagged
    return pos_tag_dict

#Lemmatize
def lemmatization(pos_tag_dic):
    lemma_dict = {}
    lemmatizer = WordNetLemmatizer()
    for key in pos_tag_dict:
        pos_tag_list = pos_tag_dict[key]
        for  word_pos in pos_tag_list:
            lemma_dict[word_pos[0]+"_"+word_pos[1]] = lemmatizer.lemmatize(word_pos[0],pos=get_wordnet_pos(word_pos[1]))
    return lemma_dict


if __name__ == "__main__":
    training_data_folder = sys.argv[1];
    files = []
    for entry in os.listdir(training_data_folder):
        if os.path.isfile(os.path.join(training_data_folder, entry)):
            files.append(os.path.join(training_data_folder, entry))
    token_dict = tokenization(files)
    pos_tag_dict = pos_tagging(token_dict)
    lemma_dict = lemmatization(pos_tag_dict)


