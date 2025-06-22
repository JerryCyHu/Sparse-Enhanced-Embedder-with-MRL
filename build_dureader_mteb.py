#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 parah_to_docindex.json + query_to_docindex.json
转换成 MTEB Retrieval 所需的 corpus / queries / qrels
"""

import sys
import json, uuid, argparse, pathlib, tqdm
from collections import defaultdict

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main():
    out_dir = pathlib.Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qrels").mkdir(exist_ok=True)

    # ---------- 1. 读入数据 ----------
    print("Loading JSON files...")
    para2doc = load_json(sys.argv[2])
    query2docs = load_json(sys.argv[3])

    # ---------- 2. 生成 corpus ----------
    print("Building corpus.jsonl ...")
    docid2paras = defaultdict(list)
    for para, docid in para2doc.items():
        docid2paras[docid].append(para)

    corpus_path = out_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as fout:
        for docid, paras in tqdm.tqdm(docid2paras.items()):
            full_doc = "\n".join(paras)
            fout.write(json.dumps({"_id": str(docid),
                                   "text": full_doc},
                                  ensure_ascii=False) + "\n")

    # ---------- 3. 生成 queries & qrels ----------
    print("Building queries.jsonl and qrels/test.tsv ...")
    queries_path = out_dir / "queries.jsonl"
    qrels_path   = out_dir / "qrels" / "test.tsv"

    with queries_path.open("w", encoding="utf-8") as q_out, \
         qrels_path.open("w", encoding="utf-8") as rel_out:
        for q_idx, (query, pos_docids) in enumerate(tqdm.tqdm(query2docs.items())):
            qid = f"q{q_idx}"
            q_out.write(json.dumps({"_id": qid, "text": query},
                                   ensure_ascii=False) + "\n")
            for docid in pos_docids:
                rel_out.write(f"{qid}\t{docid}\t1\n")

    print("✓ Done!  MTEB-ready dataset saved to:", out_dir)

if __name__ == "__main__":
    main()
