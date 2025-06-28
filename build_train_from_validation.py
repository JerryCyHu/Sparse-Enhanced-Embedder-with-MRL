#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert line‑delimited {"query","doc","rel"} records to
{"query", "pos", "neg", "pos_scores", "neg_scores", "prompt"} format.

• rel ∈ {2,3} → positive
• rel ∈ {0,1} → negative
"""

import json
from collections import defaultdict
from pathlib import Path
import random
from collections import OrderedDict  

SRC_FILE   = Path("validation.json")   # input
DEST_FILE  = Path("training.json")   # output

# ── accumulate by query ────────────────────────────────────────────────────────
grouped = defaultdict(
    lambda: {
        "query": None,
        "pos": [], "neg": [],
        "pos_scores": [], "neg_scores": [],
        "prompt": "Represent this sentence for searching relevant passages: "
    }
)

with SRC_FILE.open(encoding="utf-8") as fin:
    for line in fin:
        if not line.strip():
            continue
        item = json.loads(line)
        try:
            q, d, rel = item["query"], item["doc"], int(item["rel"])
            rel = float(rel)
        except:
            continue
        g = grouped[q]
        g["query"] = q                                   # set once
        if rel >= 2.0:                                     # positive
            g["pos"].append(d)
            g["pos_scores"].append(rel)
        else:                                            # negative
            g["neg"].append(d)
            g["neg_scores"].append(rel)
all_items = list(grouped.values())
all_items = [item for item in all_items if item['pos'] and item['neg']]
sample_size = max(1, int(0.3 * len(all_items)))
sampled_items = random.sample(all_items, sample_size)
remaining_items = [item for item in all_items if item not in sampled_items]
# ── write out ──────────────────────────────────────────────────────────────────
with DEST_FILE.open("w", encoding="utf-8") as fout:
    json.dump(sampled_items, fout, ensure_ascii=False)
print(f"Saved {len(sampled_items)} grouped records to {DEST_FILE}")

def save_rest_val(remaining_items):
    out_dir          = Path("val_rest")           # change to your target folder
    corpus_file      = out_dir / "corpus.jsonl"
    queries_file     = out_dir / "test_queries.jsonl"
    qrels_file       = out_dir / "test_qrels.jsonl"
    out_dir.mkdir(exist_ok=True)

    # ---------- 3. HELPERS ----------
    doc_lookup = OrderedDict()     # text → doc_id  (deduplicates automatically)
    global next_doc_id
    next_doc_id = 1

    def _get_doc_id(text):
        """Return existing ID or register a new one for this document text."""
        if text not in doc_lookup:
            global next_doc_id
            doc_lookup[text] = str(next_doc_id)
            next_doc_id += 1
        return doc_lookup[text]

    # ---------- 4. BUILD CORPUS / QUERIES / QRELS ----------
    queries_out, qrels_out = [], []

    for q_idx, entry in enumerate(remaining_items, start=1):
        qid = str(q_idx)
        queries_out.append({"id": qid, "text": entry["query"]})

        # positive docs → relevance 1
        for doc_text in entry["pos"]:
            docid = _get_doc_id(doc_text)
            qrels_out.append({"qid": qid, "docid": docid, "relevance": 1})

        # (optional) negative docs in corpus, but not in qrels
        for doc_text in entry["neg"]:
            _get_doc_id(doc_text)

    # ---------- 5. WRITE FILES ----------
    with corpus_file.open("w", encoding="utf‑8") as f:
        for text, docid in doc_lookup.items():
            f.write(json.dumps({"id": docid, "title": "", "text": text}, ensure_ascii=False) + "\n")

    with queries_file.open("w", encoding="utf‑8") as f:
        for q in queries_out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    with qrels_file.open("w", encoding="utf‑8") as f:
        for r in qrels_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(doc_lookup)} docs, {len(queries_out)} queries, {len(qrels_out)} qrels → {out_dir}/")
    
save_rest_val(remaining_items)