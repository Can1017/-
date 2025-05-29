# 配置预处理选项
PREPROCESS_CONFIG = {
    "stemming": True,
    "lowercase": True,
    "remove_numbers": True,
    "remove_punctuation": True
}

# 权重方案和排名函数

# tf：得分等于文档中该词出现的次数（词频）
# binary：只要文档中出现过该词，得分就是1
# logtf：得分等于log(1 + 词频)，平滑处理
# tfidf：得分是 tf × idf，tf为词频，idf为逆文档频率。出现多次的文档分数更高，且被很多文档包含的词权重会被抑制

weight_schemes = ["tf", "tfidf", "bm25"]

# desc_score：按得分从高到低排序
# #asc_docid：按文档编号从小到大排序
# #asc_docid：按文档编号从小到大排序

rank_funcs = ["desc_score"]

TOP_N = 10
data_path = './data/2k_data.json'
