#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量把 DuRetrieval‑style 数据用本地 ./bge‑m3 模型 embed 到 ChromaDB，
然后一次性评测 NDCG@10。（默认 GPU，可改成 CPU）

用法示例：
python corpus_evaluation.py \
  --data_dir ./dureader_valdata \
  --model_path ./bge-m3 \
  --chroma_dir ./chroma_m3 \
  --collection_name dureader_m3 \
  --batch_size 128
"""

import argparse, json, math, os, sys, time
from pathlib import Path
from typing import List

import chromadb
import pandas as pd
import torch
import torch.nn.functional as F
from chromadb import PersistentClient
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --------------------------- ❶ 参数 ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./val_rest", help="目录里包含 corpus.jsonl / test_queries.jsonl / test_qrels.jsonl")
    p.add_argument("--model_path", default="./acge_text_embedding")
    p.add_argument("--chroma_dir", default="./cs_acge")
    p.add_argument("--collection_name", default="doc_acge")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--top_k", type=int, default=10, help="评测时返回的文档数 (k)")
    return p.parse_args()


# --------------------------- ❷ 工具函数 ---------------------------
def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def embed_text_batch(texts: List[str], model, batch_size):
    # convert_to_numpy=True → 直接得到 np.ndarray；normalize_embeddings=True → 自动 L2 归一化
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(true_set: set, retrieved: List[str], k: int):
    rels  = [1 if doc_id in true_set else 0 for doc_id in retrieved[:k]]
    ideal = [1]*min(k, len(true_set)) + [0]*(k - min(k, len(true_set)))
    return dcg(rels) / dcg(ideal) if ideal else 0.0


# --------------------------- ❸ 主流程 ---------------------------
def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # ---------- 1. 模型 ----------
    model = SentenceTransformer(args.model_path, device=device, trust_remote_code=True)

    # ---------- 2. ChromaDB ----------
    client = PersistentClient(path=args.chroma_dir)
    collection = client.get_or_create_collection(
        name=args.collection_name, metadata={"hnsw:space": "ip"}
    )

    # 若已存在向量，跳过重建
    if collection.count() == 0:
        print("[INFO] Embedding corpus → ChromaDB")
        corpus_path = data_dir / "corpus.jsonl"
        docs = list(load_jsonl(corpus_path))
        doc_texts = [d["text"] for d in docs]
        doc_ids = [d["id"] for d in docs]

        for i in tqdm(range(0, len(docs), args.batch_size), desc="Indexing"):
            batch_texts = doc_texts[i : i + args.batch_size]
            batch_ids = doc_ids[i : i + args.batch_size]
            batch_embs = embed_text_batch(batch_texts, model, args.batch_size)
            collection.add(ids=batch_ids, embeddings=batch_embs, documents=batch_texts)
        print(f"[INFO] Index built: {collection.count()} embeddings")
    else:
        print("[INFO] Found existing index, skip embedding")

    # ---------- 3. 加载 queries & qrels ----------
    queries = list(load_jsonl(data_dir / "test_queries.jsonl"))
    qrels = list(load_jsonl(data_dir / "test_qrels.jsonl"))

    # 构建 qid -> 正例 docid set（只保留 relevance>0）
    qid2positives = {}
    for rec in qrels:
        qid = rec["qid"]
        rel = rec.get("relevance", 0)
        if rel > 0:
            qid2positives.setdefault(qid, set()).add(rec["docid"])

    # ---------- 4. 评测 ----------
    ndcg_total, valid_q = 0.0, 0
    for i in tqdm(range(0, len(queries), args.batch_size), desc="Evaluating"):
        batch = queries[i : i + args.batch_size]
        texts = [q["text"] for q in batch]
        qids = [q["id"] for q in batch]
        q_embs = embed_text_batch(texts, model, args.batch_size)

        results = collection.query(
            query_embeddings=q_embs,
            n_results=args.top_k,
            include=["documents"],
        )["ids"]  # List[List[str]]

        for qid, retrieved in zip(qids, results):
            if qid not in qid2positives:
                continue
            ndcg = ndcg_at_k(qid2positives[qid], retrieved, args.top_k)
            ndcg_total += ndcg
            valid_q += 1

    print(f"[RESULT] NDCG@{args.top_k} = {ndcg_total / valid_q:.4f}  (over {valid_q} queries)")


if __name__ == "__main__":
    main()
