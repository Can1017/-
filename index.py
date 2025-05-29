from collections import defaultdict
from preprocess import preprocess


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)

    def add(self, token, doc_id):
        self.index[token].append(doc_id)

    def dictionary_size(self):
        """返回字典（唯一token/biword）大小"""
        return len(self.index)

    def index_size(self):
        """返回索引（倒排链总长度）大小"""
        return sum(len(postings) for postings in self.index.values())


def build_single_word_index(docs, config):
    index = InvertedIndex()
    for doc_id, text in docs.items():
        tokens = preprocess(text, config)
        for token in tokens:
            index.add(token, doc_id)
    return index


def build_biword_index(docs, config):
    index = InvertedIndex()
    for doc_id, text in docs.items():
        tokens = preprocess(text, config)
        for i in range(len(tokens)-1):
            biword = tokens[i] + ' ' + tokens[i+1]
            index.add(biword, doc_id)
    return index
