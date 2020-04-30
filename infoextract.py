import os
import sys
from pprint import pprint

# nltk and spacy
import nltk
import spacy
from spacy.pipeline import merge_entities
from spacy.symbols import nsubj, VERB

# gensim and codecs
import codecs
import gensim
from  gensim import corpora
from gensim.models import CoherenceModel

nlp = spacy.load('en_core_web_sm')
nltk.download('punkt')

if __name__ == "__main__":
    files = []
    training_data_folder = sys.argv[1]
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

    # These rules reducing the data size too much

    # def extract_person_orgs(doc):
    #     person_entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]
    #     for ent in person_entities:
    #         head = ent.root.head
    #         if head.lemma_ in ["work","employment","run",
    #                            "toil","function","operate",
    #                             "perform","run","labor"]:
    #             preps = [token for token in head.children if token.dep_ == "prep"]
    #             for prep in preps:
    #                 orgs = [token for token in prep.children if token.ent_type_ == "ORG"]
    #                 if len(orgs) > 0:
    #                     corpus.add(doc)
    #                     break
    #     return doc

    # def extract_places(doc):
    #   location_entities = [ent for ent in doc.ents if ent.label_.lower() in
    #                      ["ORG","NORP","FAC","GPE", "LOC"]]
    #   if len(location_entities) > 2:
    #     for ent in location_entities:
    #         head = ent.root.head
    #         if head.lemma_ in ["be", "situate", "place", "locate"]:
    #             corpus.add(doc)
    #             break
    #   return doc

    # def extract_by_verb(doc):
    #     for possible_subject in doc:
    #         if possible_subject.dep == nsubj and possible_subject.head.pos == VERB and possible_subject.head.lemma_.lower() in ["buy","purchase","acquire"
    #                                       ,"trade", "obtain", "get", "take", "sale"]:

    #             corpus.add(doc)
    #         if possible_subject.dep == nsubj and possible_subject.head.pos == VERB and possible_subject.head.lemma_.lower() in ["work","employment","run",
    #                                                       "toil","function","operate",
    #                                                       "perform","run","labor"]:

    #             corpus.add(doc)
    #         if possible_subject.dep == nsubj and possible_subject.head.pos == VERB and possible_subject.head.lemma_.lower() in ["located","situate","locate",
    #                                                       "part"]:
    #             corpus.add(doc)
    #     return doc

    def extract_entities(doc):
        for ent in doc.ents:
            if ent.label_ == "ORG" or ent.label_ == "GPE" \
                    or ent.label_ == "LOC" or ent.label_ == "PERSON":
                head = ent.root.head;
                if head.lemma_.lower() in ["work", "employment", "run", "employ", "purchaser"
                                                                                  "toil", "function", "operate",
                                           "perform", "run", "labor", "buy", "purchase", "acquire"
                    , "trade", "obtain", "get", "take", "sale", "located",
                                           "situate", "locate", "part", "locate", "founder", "lived"
                                                                                             "headquarter", "serve"]:
                    return doc
        return nlp.make_doc("")


    def preprocessing(doc):
        # print(fc)
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


    nlp.add_pipe(preprocessing)
    nlp.add_pipe(extract_entities)
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
    bigram = gensim.models.Phrases(corpus, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[corpus], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    print(trigram_mod[bigram_mod[corpus[0]]])

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

    # Creates, which is a mapping of word IDs to words.
    words = corpora.Dictionary(corpus)
    words.filter_extremes(no_below=5, no_above=0.8)
    # Turns each document into a bag of words.
    filtered_corpus = [words.doc2bow(doc) for doc in corpus]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=filtered_corpus,
                                                id2word=words,
                                                num_topics=3,
                                                random_state=3,
                                                update_every=1,
                                                passes=100,
                                                alpha='auto',
                                                per_word_topics=True)

    pprint(lda_model.print_topics(num_words=10))
