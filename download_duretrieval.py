from datasets import load_dataset
import pandas as pd, json
import os

# --------------------------- ❶ 加载 DuRetrieval 数据 ---------------------------
# corpus  —— passages（共 100k）
# queries —— queries  （共 2k）
# 默认 config = qrels —— qrels （共 9k 对应关系）
corpus  = load_dataset("mteb/DuRetrieval", "corpus",  split="dev")
queries = load_dataset("mteb/DuRetrieval", "queries", split="dev")
qrels   = load_dataset("mteb/DuRetrieval",            split="dev")   # 默认 config = qrels

# --------------------------- ❷ 创建输出目录 ---------------------------
os.makedirs("./duretrieval_data", exist_ok=True)

# --------------------------- ❸ 写入 corpus.jsonl ---------------------------
# 格式要求：{"id": "...", "title": "", "text": ""}
with open("./duretrieval_data/corpus.jsonl", "w", encoding="utf-8") as f:
    for row in corpus:
        obj = {
            "id":    row["_id"],
            "title": row.get("title", ""),
            "text":  row.get("text", "")
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# --------------------------- ❹ 写入 test_queries.jsonl ---------------------------
# 格式要求：{"id": "...", "text": "..."}
with open("./duretrieval_data/test_queries.jsonl", "w", encoding="utf-8") as f:
    for row in queries:
        obj = {
            "id":   row["_id"],
            "text": row.get("text", "")
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# --------------------------- ❺ 写入 test_qrels.jsonl ---------------------------
# 首先把 qrels 转成 DataFrame，然后逐行写成 {"qid": "...", "docid": "...", "relevance": 1}
df_qrels = pd.DataFrame(qrels)
with open("./duretrieval_data/test_qrels.jsonl", "w", encoding="utf-8") as f:
    for _, row in df_qrels.iterrows():
        obj = {
            "qid":       row["query-id"],
            "docid":     row["corpus-id"],
            "relevance": int(row["score"])
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
