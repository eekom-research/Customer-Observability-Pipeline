from db import Model, Session, engine
from models import Tweet, ProcessedTweet, Company
from sqlalchemy import select, or_, func

import hashlib
import re  # regular expression
import spacy
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

nlp = spacy.load("en_core_web_sm")
topic_mapping_gensim_lda = {
    '0': 'Fraud',
    '1': 'Mobile App',
    '2': 'General Enquiry',
    '3': 'Customer Service Feedback',
    '4': 'Physical Branch',
    '5': 'General Enquiry',
    '6': 'Money Transfer Issues',
    '7': 'Dispense Error Issues',
    '8': 'Credit Products',
    '9': 'General Enquiry'
}

topic_mapping_sk_lda = {
    '0': 'Fraud',
    '1': 'Miscellaneous',
    '2': 'Transaction issues',
    '3': 'Card Issues',
    '4': 'Mobile App',
    '5': 'Miscellaneous',
    '6': 'Physical Branch',
    '7': 'General Enquiry',
    '8': 'Miscellaneous',
    '9': 'Dispense Error Issues'
}


def get_raw_tweets(query_limit=10):
    companies_info_query = select(Company.username)
    with Session() as session:
        companies_username = session.scalars(companies_info_query).all()

    like_conditions = [func.lower(Tweet.text).like(f'%{username}%') for username in companies_username]
    subquery = select(ProcessedTweet.id)
    subquery_condition = Tweet.id.notin_(subquery)
    query = select(Tweet).where(or_(*like_conditions)).where(subquery_condition).limit(query_limit)

    # Fetch raw tweets
    with Session() as session:
        raw_tweets = session.scalars(query).all()

    return raw_tweets


def compute_unique_primary_key(input_str):
    hashed_string = hashlib.sha256(input_str.encode('utf-8')).hexdigest()
    return hashed_string


def normalize_text(documents,
                   min_token_len=1,
                   irrelevant_pos=['ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE']):
    """
    Given text, min_token_len, and irrelevant_pos carry out preprocessing of the text
    and return a preprocessed string.

    Keyword arguments:
    documents -- (np.array[str]) the list of documents to be preprocessed
    min_token_len -- (int) min_token_length required
    irrelevant_pos -- (list) a list of irrelevant pos tags

    Returns: np.array[str] the normalized documents
    """
    normalized_documents = []

    for text in documents:
        # print(text)
        # Remove Emails
        text = re.sub(r'\S*@\S*\s?', '', text)

        # Remove extra space characters
        text = re.sub(r'\s+', ' ', text)

        # Remove distracting characters
        text = re.sub(r'''[\*\~]+''', "", text)

        doc = nlp(text)  # covert text into spacy object
        clean_text = []

        for token in doc:
            if (token.is_stop == False  # Check if it's not a stopword
                    and token.is_alpha  # Check if it's an alphanumerics char
                    and len(token) > min_token_len  # Check if the word meets minimum threshold
                    and token.pos_ not in irrelevant_pos):  # Check if the POS is in the acceptable POS tags
                lemma = token.lemma_  # Take the lemma of the word
                clean_text.append(lemma)

        clean_text = ' '.join(clean_text)  # merge list of tokens back into string
        normalized_documents.append(clean_text)  # append to list of normalized documents

    normalized_documents = np.array(normalized_documents)  # convert list of normalized documents into numpy array
    return normalized_documents


def tokenizer_func(X):
    return [word_tokenize(doc.lower()) for doc in X]


def get_topic_assignment(array, topic_mapping):
    temp_result = (max(range(len(array)), key=array.__getitem__), max(array))
    if temp_result[1] < 0.4:
        topic = 'Unclassified'
    else:
        topic = topic_mapping.get(str(temp_result[0]), 'Unclassified')
    return topic


class GensimLdaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, gensim_model, gensim_dictionary):
        self.gensim_model = gensim_model
        self.gensim_dictionary = gensim_dictionary

    def fit(self, X, y=None):
        # Since the model is already trained, we don't need to do anything here
        return self

    def transform(self, X):
        # Transform the data into the bag-of-words format
        corpus = [self.gensim_dictionary.doc2bow(doc) for doc in X]
        # Use the Gensim model to transform the data
        transformed_corpus = [self.gensim_model.get_document_topics(bow, minimum_probability=0) for bow in corpus]
        # Return the transformed data in a format suitable for scikit-learn
        # Here, we're returning the topic distribution for each document
        return [[prob for _, prob in doc_topics] for doc_topics in transformed_corpus]
