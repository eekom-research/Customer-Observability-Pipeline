import re  # regular expression
import spacy
import emoji
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

nlp = spacy.load("en_core_web_sm")


def normalize_text(documents,
                   min_token_len=1,
                   irrelevant_pos=('ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE')):
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


def tokenizer_func(x):
    return [word_tokenize(doc.lower()) for doc in x]


# define emoji removal helper function
def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")
