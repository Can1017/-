from collections import defaultdict
from preprocess import preprocess
from tqdm import tqdm
from joblib import Parallel, delayed


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)

    def add(self, token, doc_id):
        self.index[token].add(doc_id)

    def dictionary_size(self):
        return len(self.index)

    def index_size(self):
        return sum(len(postings) for postings in self.index.values())


def build_single_word_index(docs, config, n_jobs=16):
    index = InvertedIndex()
    # 并行预处理

    def process(doc_id, text):
        tokens = preprocess(text, config)
        return doc_id, tokens
    results = Parallel(n_jobs=n_jobs)(
        delayed(process)(doc_id, text) for doc_id, text in tqdm(docs.items(), desc="预处理 single-word"))
    for doc_id, tokens in results:
        for token in tokens:
            index.add(token, doc_id)
    return index


def build_biword_index(docs, config, n_jobs=16):
    index = InvertedIndex()

    def process(doc_id, text):
        tokens = preprocess(text, config)
        biwords = [tokens[i] + ' ' + tokens[i+1] for i in range(len(tokens)-1)]
        return doc_id, biwords
    results = Parallel(n_jobs=n_jobs)(
        delayed(process)(doc_id, text) for doc_id, text in tqdm(docs.items(), desc="预处理 biword"))
    for doc_id, biwords in results:
        for biword in biwords:
            index.add(biword, doc_id)
    return index
