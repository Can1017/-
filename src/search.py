from collections import defaultdict
from preprocess import preprocess
import math


def get_doc_lengths(index):
    # 统计每个文档的长度（即该文档所有token的数量）
    doc_lengths = {}
    for postings in index.index.values():
        for doc_id in postings:
            doc_lengths[doc_id] = doc_lengths.get(doc_id, 0) + 1
    return doc_lengths


def compute_idf(index, N):
    idf = {}
    for term, postings in index.index.items():
        df = len(set(postings))
        idf[term] = math.log((N + 1) / (df + 1)) + 1  # 加1平滑
    return idf


def search(query, index, config, top_n=10, scheme="tf", idf_dict=None, rank_func="desc_score", doc_lengths=None, avg_dl=None, k1=1.5, b=0.75):
    tokens = preprocess(query, config)
    doc_scores = defaultdict(float)
    for token in tokens:
        postings = index.index.get(token, [])
        if scheme == "tf":
            for doc_id in postings:
                doc_scores[doc_id] += 1
        elif scheme == "binary":
            for doc_id in set(postings):
                doc_scores[doc_id] += 1
        elif scheme == "logtf":
            tf_count = defaultdict(int)
            for doc_id in postings:
                tf_count[doc_id] += 1
            for doc_id, tf in tf_count.items():
                doc_scores[doc_id] += math.log(1 + tf)
        elif scheme == "tfidf":
            if idf_dict is None:
                raise ValueError("idf_dict required for tfidf scheme")
            tf_count = defaultdict(int)
            for doc_id in postings:
                tf_count[doc_id] += 1
            for doc_id, tf in tf_count.items():
                doc_scores[doc_id] += tf * idf_dict.get(token, 0)
        elif scheme == "bm25":
            if idf_dict is None or doc_lengths is None or avg_dl is None:
                raise ValueError(
                    "idf_dict, doc_lengths, avg_dl required for bm25 scheme")
            tf_count = defaultdict(int)
            for doc_id in postings:
                tf_count[doc_id] += 1
            for doc_id, tf in tf_count.items():
                dl = doc_lengths.get(doc_id, avg_dl)
                idf = idf_dict.get(token, 0)
                score = idf * (tf * (k1 + 1)) / \
                    (tf + k1 * (1 - b + b * dl / avg_dl))
                doc_scores[doc_id] += score
        else:
            for doc_id in postings:
                doc_scores[doc_id] += 1
    if len(tokens) > 1:
        doc_sets = [set(index.index.get(token, [])) for token in tokens]
        common_docs = set.intersection(*doc_sets) if doc_sets else set()
        doc_scores = {doc_id: score for doc_id,
                      score in doc_scores.items() if doc_id in common_docs}
    # 排名方式
    if rank_func == "desc_score":
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    elif rank_func == "asc_score":
        ranked = sorted(doc_scores.items(), key=lambda x: x[1])
    elif rank_func == "asc_docid":
        ranked = sorted(doc_scores.items(), key=lambda x: x[0])
    else:
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def biword_search(phrase, biword_index, config):
    tokens = preprocess(phrase, config)
    if len(tokens) < 2:
        return []
    biwords = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    doc_sets = [set(biword_index.index.get(bw, [])) for bw in biwords]
    if not doc_sets:
        return []
    common_docs = set.intersection(*doc_sets)
    # 分数可以设为biword命中数
    return [(doc_id, len(biwords)) for doc_id in common_docs]
