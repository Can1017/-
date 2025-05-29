import pandas as pd
import nltk
from config import *
from index import build_single_word_index, build_biword_index
from preprocess import preprocess
from search import biword_search, search, compute_idf, get_doc_lengths
from evaluate import evaluate


def load_data(filepath):
    df = pd.read_json(filepath, lines=True)
    docs = dict(zip(df.index, df['text']))
    return docs


def get_snippet(doc_text, query_terms, snippet_len=50, phrases=None):
    text_lower = doc_text.lower()
    # 1. 优先查找短语
    if phrases:
        for phrase in phrases:
            idx = text_lower.find(phrase.lower())
            if idx != -1:
                start = max(0, idx - snippet_len // 2)
                end = min(len(doc_text), idx + len(phrase) + snippet_len // 2)
                snippet = doc_text[start:end].replace('\n', ' ')
                # 绿色高亮
                colored = f"\033[91m{doc_text[idx:idx+len(phrase)]}\033[0m"
                snippet = snippet.replace(
                    doc_text[idx:idx+len(phrase)], colored)
                return snippet
    # 2. 查找单词
    for term in query_terms:
        idx = text_lower.find(term.lower())
        if idx != -1:
            start = max(0, idx - snippet_len // 2)
            end = min(len(doc_text), idx + len(term) + snippet_len // 2)
            snippet = doc_text[start:end].replace('\n', ' ')
            colored = f"\033[91m{doc_text[idx:idx+len(term)]}\033[0m"
            snippet = snippet.replace(
                doc_text[idx:idx+len(term)], colored)
            return snippet
    return doc_text[:snippet_len] + ("..." if len(doc_text) > snippet_len else "")


def get_all_terms_relevant_docs(query_terms, docs):
    relevant = set()
    for doc_id, text in docs.items():
        text_lower = text.lower()
        if all(term.lower() in text_lower for term in query_terms):
            relevant.add(doc_id)
    return relevant


def get_head_relevant_docs(query_terms, docs, head_n=10):
    relevant = set()
    for doc_id, text in docs.items():
        tokens = text.lower().split()[:head_n]
        if all(term.lower() in tokens for term in query_terms):
            relevant.add(doc_id)
    return relevant


def get_strict_relevant_docs(query_terms, docs, window=10):
    relevant = set()
    for doc_id, text in docs.items():
        tokens = text.lower().split()
        positions = []
        for term in query_terms:
            if term.lower() in tokens:
                positions.append(tokens.index(term.lower()))
            else:
                break
        if len(positions) == len(query_terms) and max(positions) - min(positions) <= window:
            relevant.add(doc_id)
    return relevant


def print_config(config):
    print("当前预处理配置：", ", ".join([f"{k}={v}" for k, v in config.items()]))


def main():
    nltk.download('punkt')
    docs = load_data(data_path)
    N = len(docs)

    # 可动态修改的预处理配置
    config = PREPROCESS_CONFIG.copy()
    TOP_N = 10

    print("欢迎使用信息检索系统。")
    print("支持命令: set [option] [True/False]，如 set stemming False")
    print("支持命令: show config (查看当前配置)")
    print("支持命令: N=10 (设置返回结果数)")
    print("支持命令: reload (重新构建索引) ")
    print("支持命令: exit (退出系统)")
    print("直接输入查询 (支持短语), 可查看不同权重和排名下的前 N 个结果")
    print_config(config)

    # 构建索引
    single_index = build_single_word_index(docs, config)
    biword_index = build_biword_index(docs, config)
    idf_dict = compute_idf(single_index, N)
    doc_lengths = get_doc_lengths(single_index)
    avg_dl = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1

    import re

    def parse_query(query):
        phrases = re.findall(r'"([^"]+)"', query)
        query_wo_phrases = re.sub(r'"[^"]+"', '', query)
        free_words = query_wo_phrases.strip()
        return phrases, free_words

    while True:
        query = input("\n请输入查询或命令(支持短语, 输入exit退出):").strip()
        if query.lower() == "exit":
            break
        if query.lower().startswith("set "):
            # 动态修改预处理配置
            try:
                _, key, value = query.split()
                if key in config:
                    config[key] = value.lower() == "true"
                    print(f"已设置 {key} = {config[key]}")
                    # 自动重建索引
                    print("配置已更改，正在重新构建索引...")
                    single_index = build_single_word_index(docs, config)
                    biword_index = build_biword_index(docs, config)
                    idf_dict = compute_idf(single_index, N)
                    print("索引已重建。")
                else:
                    print("无效配置项。可选：", list(config.keys()))
            except Exception:
                print("命令格式错误，应为 set [option] [True/False]")
            continue
        if query.lower() == "show config":
            print_config(config)
            continue
        if query.lower().startswith("n="):
            try:
                n_val = int(query.split("=")[1])
                TOP_N = n_val
                print(f"已设置TOP_N = {TOP_N}")
            except Exception:
                print("N值设置错误, 应为 N=数字")
            continue
        if query.lower() == "reload":
            print("重新构建索引...")
            single_index = build_single_word_index(docs, config)
            biword_index = build_biword_index(docs, config)
            idf_dict = compute_idf(single_index, N)
            print("索引已重建。")
            continue
        if not query:
            continue

        phrases, free_words = parse_query(query)
        all_terms = []
        for phrase in phrases:
            all_terms.extend(phrase.strip().split())
        if free_words:
            all_terms.extend(free_words.strip().split())

        print(f"\n查询: {query}")
        print_config(config)
        print(
            f"字典大小：{single_index.dictionary_size()}，索引大小：{single_index.index_size()}")

        # 对每种权重方案和排名函数组合，展示前N个结果
        for scheme in weight_schemes:
            for rank_func in rank_funcs:
                doc_scores = {}
                # 1. 处理短语部分（biword索引）
                for phrase in phrases:
                    tokens = preprocess(phrase, config)
                    if len(tokens) == 1:
                        # 单个词，走单词索引
                        if scheme == "bm25":
                            results = search(
                                tokens[0], single_index, config, TOP_N, scheme, idf_dict, rank_func, doc_lengths, avg_dl)
                        else:
                            results = search(
                                tokens[0], single_index, config, TOP_N, scheme, idf_dict if scheme == "tfidf" else None, rank_func)
                        for doc_id, score in results:
                            doc_scores[doc_id] = doc_scores.get(
                                doc_id, 0) + score
                    else:
                        # 多词短语，走biword索引
                        if len(preprocess(phrase, config)) > 1:
                            results = biword_search(
                                phrase, biword_index, config)
                            for doc_id, score in results:
                                doc_scores[doc_id] = doc_scores.get(
                                    doc_id, 0) + score
                # 2. 处理自由词部分（单词索引）
                if free_words:
                    # 一次性传入所有自由词
                    results = search(free_words, single_index, config, TOP_N, scheme,
                                     idf_dict if scheme == "tfidf" else idf_dict if scheme == "bm25" else None,
                                     rank_func,
                                     doc_lengths if scheme == "bm25" else None,
                                     avg_dl if scheme == "bm25" else None)
                    for doc_id, score in results:
                        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
                # 3. 排序输出
                if rank_func == "desc_score":
                    ranked = sorted(doc_scores.items(),
                                    key=lambda x: x[1], reverse=True)[:TOP_N]
                elif rank_func == "asc_score":
                    ranked = sorted(doc_scores.items(),
                                    key=lambda x: x[1])[:TOP_N]
                elif rank_func == "asc_docid":
                    ranked = sorted(doc_scores.items(),
                                    key=lambda x: x[0])[:TOP_N]
                else:
                    ranked = sorted(doc_scores.items(),
                                    key=lambda x: x[1], reverse=True)[:TOP_N]
                print(f"\n[权重方案: {scheme} | 排序方式: {rank_func}]")
                print("Rank\tScore\tDocID\tSummary Snippet")
                retrieved_docids = []
                for rank, (doc_id, score) in enumerate(ranked, 1):
                    doc_text = docs[doc_id]
                    snippet = get_snippet(doc_text, all_terms, phrases=phrases)
                    print(f"{rank}\t{score}\t{doc_id}\t{snippet}")
                    retrieved_docids.append(doc_id)

                # 计算并打印评估指标
                rel_docs = get_all_terms_relevant_docs(all_terms, docs)
                precision, recall, f1 = evaluate(retrieved_docids, rel_docs)
                print(
                    f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")


if __name__ == "__main__":
    main()