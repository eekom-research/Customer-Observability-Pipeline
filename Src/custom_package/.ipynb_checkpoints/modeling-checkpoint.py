from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

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
    '0': 'Mobile Banking',
    '1': 'Accounted Related Queries',
    '2': 'General Complaints',
    '3': 'General Enquiry',
    '4': 'General Enquiry',
    '5': 'General Enquiry',
    '6': 'Transaction Related Issues',
    '7': 'Mobile Banking',
    '8': 'Transaction Related Issue',
    '9': 'Fraud'
}

topic_mapping_sk_full_lda = {
    '0': 'General Inquiries',
    '1': 'General Complaints',
    '2': 'Transactions - Value Added Services',
    '3': 'Electronic Banking - Transaction Errors',
    '4': 'Electronic Banking - General',
    '5': 'Transaction Issues - Card Services',
    '6': 'Account Related Issues',
    '7': 'Electronic Banking - Complaints & Fraud Reports',
    '8': 'Transaction Issues - General',
    '9': 'Transaction Issues - Agent Banking'
}


# Function to get and map sentiment
def get_pos_sentiment_proba(text):
    result = text
    # Map the model's label to a more descriptive term
    pos_sentiment_proba = ((result[-1].get('score', np.nan) - result[-3].get('score', np.nan)) + 1) / 2
    return pos_sentiment_proba


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
