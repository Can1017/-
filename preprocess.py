import re
from nltk.stem import PorterStemmer


def preprocess(text, config):
    if config["lowercase"]:
        text = text.lower()
    if config["remove_numbers"]:
        text = re.sub(r'\d+', '', text)
    if config["remove_punctuation"]:
        text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    if config["stemming"]:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    return tokens
