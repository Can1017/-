def evaluate(retrieved, relevant):
    """
    retrieved: 检索返回的文档id列表
    relevant:  标准相关文档id集合
    """
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    tp = len(retrieved_set & relevant_set)
    precision = tp / len(retrieved) if retrieved else 0.0
    recall = tp / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1
