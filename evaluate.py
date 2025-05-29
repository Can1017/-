def evaluate(results, relevant_docs):
    tp = len(set(results) & set(relevant_docs))
    precision = tp / len(results) if results else 0
    recall = tp / len(relevant_docs) if relevant_docs else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision+recall) else 0
    return precision, recall, f1
