import os
import sys


# nltk and spacy
import nltk
import spacy

from spacy.symbols import nsubj, VERB

# gensim and codecs
import codecs
import gensim
from  gensim import corpora
from gensim.models import CoherenceModel
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

nlp = spacy.load('en_core_web_sm')
nltk.download('punkt')
nltk.download('wordnet')

if __name__ == "__main__":
    files = []
    training_data_folder = sys.argv[1]
    test_data_folder = sys.argv[2]
    for entry in os.listdir(training_data_folder):
        if os.path.isfile(os.path.join(training_data_folder, entry)):
            files.append(os.path.join(training_data_folder, entry))
    corpus_raw = u""
    for file in files:
        with codecs.open(file, "r", "utf-8") as text:
            corpus_raw += text.read()

    sentence_tokens = {}
    sentence_pos = {}
    sentence_lemma = {}
    sentence_NER = {}
    nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')


    def extract_entities(doc):
        ent_count = 0
        entities = [ent for ent in doc.ents]
        token_list = []
        for token in doc:
            token_list.append(token)
        for ent in entities:
            if ent.label_ == "PERSON":
                for token in doc:
                    head = ent.root.head
                    if token.lemma_.lower() in ["work", "employment", "run",
                                                "toil", "function", "operate",
                                                "perform", "run", "labor", "employ", "founder",
                                                "serve"]:
                        # tokens = [token for token in head.children ]
                        # for word in tokens:
                        #   if word.ent_type_ in ["ORG","NORP","FAC","GPE"]:
                        for ent in entities:
                            if ent.label_.lower() in ["ORG", "NORP", "FAC", "GPE"]:
                                return doc

                    if head.lemma_ in ["work", "employment", "run",
                                       "toil", "function", "operate",
                                       "perform", "run", "labor"]:
                        preps = [token for token in head.children if token.dep_ == "prep"]
                        for prep in preps:
                            orgs = [token for token in prep.children if token.ent_type_ == "ORG"]
                            if len(orgs) > 0:
                                return doc

            if ent.label_ in ["ORG", "GPE", "LOC", "PERSON", "NORP", "FAC"]:
                for token in doc:
                    if token.lemma_.lower() in ["purchaser",
                                                "buy", "purchase", "acquire"
                        , "trade", "obtain", "get", "take", "sale"]:
                        return doc
        for i in range(len(token_list) - 1):
            if token_list[i].ent_type_ in ["ORG", "NORP", "FAC", "GPE"]:
                if token_list[i + 1].is_punct or token_list[i + 1].ent_type_ in ["ORG", "NORP", "FAC", "GPE"]:
                    return doc
            token = token_list[i]
            if token.dep == nsubj and token.head.pos == VERB and token.head.lemma_.lower() in ["purchaser", "buy",
                                                                                               "purchase", "acquire"
                , "trade", "obtain", "get",
                                                                                               "take", "sale", "work",
                                                                                               "employment", "run",
                                                                                               "toil", "function",
                                                                                               "operate",
                                                                                               "perform", "run",
                                                                                               "labor", "located",
                                                                                               "situate",
                                                                                               "locate", "part"]:
                return doc

        location_entities = [ent for ent in doc.ents if ent.label_.lower() in
                             ["ORG", "NORP", "FAC", "GPE", "LOC"]]
        if len(location_entities) > 2:
            for ent in location_entities:
                head = ent.root.head
                if head.lemma_ in ["be", "situate", "place", "locate", "headquarter", "located",
                                   "situate", "locate", "part", "lived", "live",
                                   "headquarter"]:
                    return doc

        return nlp.make_doc("")


    def preprocessing(doc):
        token_list = []
        pos_list = []
        ner_list = []
        lemma_list = []

        for token in doc:
            token_list.append(token.text)
            pos_list.append(token.pos_)
            if token.pos_ == "VERB":
                lemma_list.append(token.lemma_)

        for ent in doc.ents:
            ner_list.append(ent.label_)
        sentence_tokens[doc.text] = token_list
        sentence_pos[doc.text] = pos_list
        sentence_NER[doc.text] = ner_list
        sentence_lemma[doc.text] = lemma_list
        return doc


    def lemmatizer(doc):
        # This takes in a doc of tokens from the NER and lemmatizes them.
        # Pronouns (like "I" and "you" get lemmatized to '-PRON-', so I'm removing those.
        doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
        doc = u' '.join(doc)
        return nlp.make_doc(doc)


    def remove_stopwords(doc):
        # This will remove stopwords and punctuation.
        # Use token.text to return strings, which we'll need for Gensim.
        doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
        return doc


    set_synset = set()
    hyponym_set = set()
    hypernym_set = set()


    # predef_synset  = set()

    # predef = nlp("work employment run employ toil function operate perform run labor buy purchase purchaser acquire trade obtain get take sale located situate locate part locate founder lived headquarter serve")
    # for token in predef:
    #   for synset in token._.wordnet.synsets():
    #     predef_synset.add(synset)

    def get_synonyms(doc):
        for token in doc:
            for synset in token._.wordnet.synsets():
                set_synset.add(synset)
                for hyponym in synset.hyponyms():
                    hyponym_set.add(hyponym)
                for hypernym in synset.hypernyms():
                    hypernym_set.add(hypernym)
        return doc


    nlp.add_pipe(extract_entities)
    nlp.add_pipe(get_synonyms)
    nlp.add_pipe(preprocessing)
    nlp.add_pipe(lemmatizer)
    nlp.add_pipe(remove_stopwords, name="stopwords", last=True)

    corpus = []
    sentences = nltk.sent_tokenize(corpus_raw)
    # corpus = list(nlp.pipe(sentences))
    for sentence in sentences:
        doc = nlp(sentence)
        if doc:
            corpus.append(doc)

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(corpus)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[corpus])

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    print(trigram_mod[bigram_mod[corpus[0]]])

    length = len(corpus)
    triagram_corpus = []
    for i in range(length):
        triagram_corpus.append(trigram_mod[bigram_mod[corpus[i]]])

    #  Pinting lemma of  verbs in the corpus
    i = 0
    for key, value in sentence_lemma.items():
        if i < 40:
            i += 1
            print(key, value)

    # printing tokens in corpus
    i = 0
    for key, value in sentence_tokens.items():
        if i < 40:
            i += 1
            print(key, value)

    # printing pos tags in corpus
    i = 0
    for key, value in sentence_pos.items():
        if i < 40:
            i += 1
            print(key, value)

    # printing NER dictionary
    i = 0
    for key, value in sentence_NER.items():
        if i < 40:
            i += 1
            print(key, value)

    # creates a map of word and count
    words = corpora.Dictionary(triagram_corpus)
    words.filter_extremes(no_below=5, no_above=0.8)
    # Turns each document into a bag of words.
    bow_corpus = [words.doc2bow(doc) for doc in triagram_corpus]

    bow_doc_43 = bow_corpus[43]
    for i in range(len(bow_doc_43)):
        print("Word {} (\"{}\") appears {} time.".format(bow_doc_43[i][0],
                                                         words[bow_doc_43[i][0]],
                                                         bow_doc_43[i][1]))

    from gensim import corpora, models

    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    # from pprint import pprint
    # for doc in corpus_tfidf:
    #     pprint(doc)
    #     break
    #  Model using Bag Of Words.
    lda_model = gensim.models.ldamodel.LdaModel(corpus=bow_corpus,
                                                id2word=words,
                                                num_topics=3,
                                                random_state=3,
                                                update_every=1,
                                                passes=100,
                                                alpha='auto',
                                                per_word_topics=True)
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # Model using TFIDF
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=3, id2word=words, passes=2, workers=4)
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))

    test_files = []
    for entry in os.listdir(test_data_folder):
        if os.path.isfile(os.path.join(test_data_folder, entry)):
            test_files.append(os.path.join(test_data_folder, entry))
    corpus_test = u""
    for file in test_files:
        with codecs.open(file, "r", "utf-8") as text:
            corpus_test += text.read()

    test_corpus = []
    test_sentences = nltk.sent_tokenize(corpus_test)
    # corpus = list(nlp.pipe(sentences))
    for sentence in test_sentences:
        doc = nlp(sentence)
        if doc:
            test_corpus.append(doc)