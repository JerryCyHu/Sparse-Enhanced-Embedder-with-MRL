from collections.abc import Iterable
import os
from datasets import Dataset
import json, random
import torch
import torch.nn as nn
from sentence_transformers import util
import torch.nn.functional as F
from sentence_transformers import (
    SentenceTransformer,
)
from torch import Tensor
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding
from dataclasses import dataclass
import torch.nn.init as init

NEG_NUM = 4

class DenseSparseModel(nn.Module):
    """
    Wraps a SentenceTransformer (dense encoder) and adds a sparse‐BOW head.
    """
    def __init__(self, st_model: SentenceTransformer, keep_special=False):
        super().__init__()
        self.st_model = st_model
        self.device = self.st_model.device
        self.model_card_data = self.st_model.model_card_data

        # Expose methods the Trainer expects:
        self.tokenize = self.st_model.tokenize
        self.get_sentence_embedding_dimension = (
            self.st_model.get_sentence_embedding_dimension
        )

        # Build sparse head
        self.config = st_model[0].auto_model.config
        h_size = self.config.hidden_size  # e.g. 768

        self.sparse_linear = nn.Linear(h_size, 1, bias=False)
        nn.init.kaiming_uniform_(
            self.sparse_linear.weight,
            a=0,
            mode="fan_in",
            nonlinearity="relu"
        )

        self.vocab_size = self.st_model.tokenizer.vocab_size
        self.tokenizer = self.st_model.tokenizer
        self.keep_special = keep_special

    def _dense_embedding(self, last_hidden_state, attention_mask):
        """Use the pooling method to get the dense embedding.

        Args:
            last_hidden_state (torch.Tensor): The model output's last hidden state.
            attention_mask (torch.Tensor): Mask out padding tokens during pooling.

        Raises:
            NotImplementedError: Specified pooling method not implemented.

        Returns:
            torch.Tensor: The dense embeddings.
        """
        return last_hidden_state

    def _sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = True):
        # hidden_state: [batch, seq, hsize]
        # input_ids:     [batch, seq]
        # 1) 线性投影得到每个 token 的权重 [batch, seq, 1]
        d = hidden_state.size(-1)              # 当前 Matryoshka 维度
        # 关键行：只取前 d 列权重做点积
        token_scores = F.linear(hidden_state, self.sparse_linear.weight[:, :d], bias=None) 
        token_weights = torch.relu(token_scores).squeeze(-1)
        # 2) 直接生成 [batch, vocab_size] 并做 amax 聚合
        sparse = torch.zeros(
            input_ids.size(0), self.vocab_size,
            dtype=token_weights.dtype,
            device=token_weights.device
        )
        sparse = sparse.scatter_reduce(
            dim=1,
            index=input_ids,    # shape [batch, seq]
            src=token_weights,  # shape [batch, seq]
            reduce="amax",
            include_self=True
        )
        # 3) 屏蔽特殊 token
        mask_idx = torch.tensor([
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id
        ], device=sparse.device, dtype=torch.long)

        # 非就地清零，把新张量赋回 sparse
        sparse = sparse.index_fill(dim=1, index=mask_idx, value=0.0)
        return sparse

    def _encode(self, features):
        """Helper function to encode using input features.

        Args:
            features (Union[list, dict]): Features feed to the model.

        Returns:
            torch.Tensor: Dense embedding.
            torch.Tensor: Sparce embedding.
            torch.Tensor: Colbert vector.
        """
        dense_vecs, sparse_vecs = None, None
        last_hidden_state = self.forward(features)
        dense_vecs = self._dense_embedding(last_hidden_state['sentence_embedding'], features['attention_mask'])
        sparse_vecs = self._sparse_embedding(last_hidden_state['token_embeddings'], features['input_ids'])
        dense_vecs = F.normalize(dense_vecs, dim=-1)
        #sparse_vecs = F.normalize(sparse_vecs, dim=-1)
        return dense_vecs, sparse_vecs

    def encode(self, features):
        """Encode and get the embedding.

        Args:
            features (Union[list, dict]): Features feed to the model.

        Returns:
            torch.Tensor: Dense embeddings.
            torch.Tensor: Sparce embeddings.
            torch.Tensor: Colbert vectors.
        """
        if features is None:
            return None

        if not isinstance(features, list):
            dense_vecs, sparse_vecs = self._encode(features)
        else:
            all_dense_vecs, all_sparse_vecs = [], []
            for sub_features in features:
                dense_vecs, sparse_vecs = self._encode(sub_features)
                all_dense_vecs.append(dense_vecs)
                all_sparse_vecs.append(sparse_vecs)

            dense_vecs = torch.cat(all_dense_vecs, 0)
            sparse_vecs = torch.cat(all_sparse_vecs, 0)

        return dense_vecs.contiguous(), sparse_vecs.contiguous()

    def forward(self, *args, **kwargs):
        # Trainer expects forward() → dense embeddings
        return self.st_model(*args, **kwargs)


    def save_pretrained(self, output_dir: str, **kwargs):
        """
        Hugging Face Trainer helper.
        Saves:
        • the wrapped SentenceTransformer
        • the sparse head weights
        • a tiny JSON telling `from_pretrained` how to rebuild
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1) 从 kwargs 中取出 safe_serialization，默认为 False
        safe_serialization = kwargs.pop("safe_serialization", False)

        # 2) delegate to SentenceTransformer（透传 safe_serialization）
        dense_path = os.path.join(output_dir, "dense_model")
        self.st_model.save(dense_path, safe_serialization=safe_serialization)

        # 3) save sparse head
        torch.save(
            self.sparse_linear.state_dict(),
            os.path.join(output_dir, "sparse_linear.pt")
        )

        # 4) meta info so we can reload
        meta = {
            "keep_special": getattr(self, "keep_special", False)
        }
        with open(os.path.join(output_dir, "densesparse_meta.json"), "w") as f:
            json.dump(meta, f)


    @classmethod
    def from_pretrained(cls, output_dir: str, device=None, **kwargs):
        """
        Mirrors `save_pretrained`.
        """

        # 1) load dense model
        st_model = SentenceTransformer(os.path.join(output_dir, "dense_model"),
                                       device=device)

        # 2) (re‑)create wrapper
        with open(os.path.join(output_dir, "densesparse_meta.json")) as f:
            meta = json.load(f)
        obj = cls(st_model, keep_special=meta["keep_special"])

        # 3) load sparse head weights
        state = torch.load(os.path.join(output_dir, "sparse_linear.pt"),
                           map_location=device or "cpu")
        obj.sparse_linear.load_state_dict(state)
        obj.to(device or st_model.device)

        return obj


# ─────────────────── Custom InfoNCE Loss (batched) ─────────────────────

class CustomInfoNCELoss(nn.Module):
    """
    Batched InfoNCE:  L = -log( exp(sim(q,p_pos)/τ) / [exp(sim(q,p_pos)/τ) + Σ exp(sim(q,n)/τ)] ).

    Expects that Trainer’s collator has tokenized:
      "query"  → "query_input_ids", "query_attention_mask"
      "pos_doc"→ "pos_doc_input_ids", "pos_doc_attention_mask"
      "neg_docs"→ "neg_docs_input_ids", "neg_docs_attention_mask"
      where neg_docs_input_ids is shape [B, N_neg, L_neg].
    """
    def __init__(self, model: SentenceTransformer, temperature: float):
        super().__init__()
        self.st_model = model
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=-1)
        self.ce  = nn.CrossEntropyLoss()
        self.similarity_fct = util.cos_sim
        self.scale = 20.0

    def _compute_similarity(self, q_reps, p_reps):
        """Computes the similarity between query and passage representations using inner product.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: The computed similarity matrix.
        """
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def compute_dense_score(self, q_reps, p_reps):
        """Compute the dense score.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: The computed dense scores, adjusted by temperature.
        """
        scores = self._compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def compute_sparse_score(self, q_reps, p_reps):
        """Compute the sparse score.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: The computed sparse scores, adjusted by temperature.
        """
        scores = self._compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def get_local_score(self, q_reps, p_reps, all_scores):
        """Get the local score of queries and passages.

        Args:
            q_reps (torch.Tensor): Queries representations.
            p_reps (torch.Tensor): Passages rerpresentations.
            all_scores (torch.Tensor): All the query-passage scores computed.

        Returns:
            torch.Tensor: Local scores to compute loss.
        """
        group_size = p_reps.size(0) // q_reps.size(0)
        indices = torch.arange(0, q_reps.size(0), device=q_reps.device) * group_size
        specific_scores = []
        for i in range(group_size):
            specific_scores.append(
                all_scores[torch.arange(q_reps.size(0), device=q_reps.device), indices + i]
            )
        return torch.stack(specific_scores, dim=1).view(q_reps.size(0), -1)

    def distill_loss(self, kd_loss_type, teacher_targets, student_scores, group_size=None):
        """Compute the distillation loss.

        Args:
            kd_loss_type (str): Type of knowledge distillation loss, supports "kl_div" and "m3_kd_loss".
            teacher_targets (torch.Tensor): Targets from the teacher model.
            student_scores (torch.Tensor): Score of student model.
            group_size (int, optional): Number of groups for . Defaults to ``None``.

        Raises:
            ValueError: Invalid kd_loss_type

        Returns:
            torch.Tensor: A scalar of computed distillation loss.
        """
        if kd_loss_type == 'kl_div':
            # teacher_targets: (batch_size, group_size) / (world_size * batch_size, group_size)
            # student_scores: (batch_size, group_size) / (world_size * batch_size, group_size)
            
            return - torch.mean(
                torch.sum(torch.log_softmax(student_scores, dim=-1) * teacher_targets, dim=-1)
            )

            # log_p = torch.log_softmax(student_scores, dim=-1)
            
            # # print("log_p forward min/max:", log_p.min().item(), log_p.max().item(), flush=True)
            # # print("student forward min/max:", student_scores.min().item(), student_scores.max().item(), flush=True)
            # # student_scores.register_hook(dbg("grad->student_scores"))
            # # log_p.register_hook(dbg("grad->log_p"))
            
            # loss  = F.kl_div(log_p, teacher_targets, reduction="batchmean", log_target=False)

            # return loss  
    def compute_local_score(self, q_reps, p_reps, compute_score_func, **kwargs):
        """Compute the local score of queries and passages.

        Args:
            q_reps (torch.Tensor): Queries representations.
            p_reps (torch.Tensor): Passages rerpresentations.
            compute_score_func (function, optional): Function to compute score. Defaults to ``None``, which will use the
                :meth:`self.compute_score`.

        Returns:
            torch.Tensor: Local scores to compute loss.
        """
        all_scores = compute_score_func(q_reps, p_reps, **kwargs)
        loacl_scores = self.get_local_score(q_reps, p_reps, all_scores)
        return loacl_scores

    def compute_loss(self, scores, target):
        """Compute the loss using cross entropy.

        Args:
            scores (torch.Tensor): Computed score.
            target (torch.Tensor): The target value.

        Returns:
            torch.Tensor: The computed cross entropy loss.
        """
        return self.ce(scores, target)

    def _compute_no_in_batch_neg_loss(self, q_reps, p_reps, teacher_targets=None, compute_score_func=None, **kwargs):
        """
        Compute loss when using no in-batch negatives and no cross-device negatives
        """
        group_size = p_reps.size(0) // q_reps.size(0)

        local_scores = self.compute_local_score(q_reps, p_reps, compute_score_func, **kwargs)   # (batch_size, group_size)

        local_targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long) # (batch_size)
        loss = self.compute_loss(local_scores, local_targets)

        return local_scores, loss
    
    def _compute_in_batch_neg_loss(self, q_reps, p_reps, teacher_targets=None, compute_score_func=None, **kwargs):
        """
        Compute loss when only using in-batch negatives
        """
        group_size = p_reps.size(0) // q_reps.size(0)

        scores = compute_score_func(q_reps, p_reps, **kwargs)   # (batch_size, batch_size * group_size)

        idxs = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long)
        targets = idxs * group_size # (batch_size)
        loss = self.compute_loss(scores, targets)

        return scores, loss
    
    def forward(self, sentence_features, labels: Tensor) -> Tensor:
        queries = sentence_features[0]['input_ids']
        passages = sentence_features[1]['input_ids']
        q_dense_vecs, q_sparse_vecs = self.st_model.encode(queries)  # (batch_size, dim)
        p_dense_vecs, p_sparse_vecs = self.st_model.encode(passages) # (batch_size * group_size, dim)

        teacher_targets = None#no teacher score

        compute_loss_func = self._compute_in_batch_neg_loss

        # dense loss
        dense_scores, loss = compute_loss_func(
            q_dense_vecs, p_dense_vecs, teacher_targets=teacher_targets,
            compute_score_func=self.compute_dense_score
        )

        # sparse loss
        sparse_scores, sparse_loss = compute_loss_func(
            q_sparse_vecs, p_sparse_vecs, teacher_targets=teacher_targets,
            compute_score_func=self.compute_sparse_score
        )
        # group_size = p_dense_vecs.size(0) // q_dense_vecs.size(0)
        # indices = torch.arange(0, q_dense_vecs.size(0), device=q_dense_vecs.device) * group_size
        # p_dense_vecs = p_dense_vecs[indices, :]
        # ensemble loss
        ensemble_scores = dense_scores + 0.3*sparse_scores 
        local_targets = torch.zeros(ensemble_scores.size(0), device=ensemble_scores.device, dtype=torch.long)
        ensemble_loss = self.compute_loss(ensemble_scores, local_targets)

        loss = (loss + ensemble_loss + 0.1 * sparse_loss) / 3

        self_teacher_targets = torch.softmax(ensemble_scores.detach(), dim=-1)

        dense_self_distill_loss = self.distill_loss("kl_div", self_teacher_targets, dense_scores)
        sparse_self_distill_loss = self.distill_loss("kl_div", self_teacher_targets, sparse_scores)

        loss += (dense_self_distill_loss + 0.1 * sparse_self_distill_loss) / 2
        loss = loss / 2

        return loss



# ────────────────────────── Dataset Loading ─────────────────────────────
def sample_list_items(lst: list, k: int = 3):
    """
    Randomly sample k elements from the list lst.
    If len(lst) < k, include all elements and then pad with None up to length k.
    Returns a list of length k containing either sampled elements or None.
    """
    n = len(lst)

    if n >= k:
        # If we have at least k elements, pick k distinct ones
        chosen = random.sample(lst, k)
    else:
        # Otherwise, take everything and pad the remainder with None
        chosen = lst.copy()
        chosen.extend(['None'] * (k - n))

    return chosen
def load_triplets(json_path):
    # Build a HuggingFace Dataset with columns:
    #   "query" (string), "pos_doc" (string), "neg_docs" (list of strings)
    rows = {
    "query": [],
    "pos_doc": [],
    }
    for i in range(1, NEG_NUM+1):
        rows[f"neg_{i}"] = []
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
        for j in data:
            q = j["query"]
            for pos in j["pos"]:
                rows["query"].append(q)
                rows["pos_doc"].append(pos)
                neg_list = sample_list_items(j["neg"], k = NEG_NUM)
                for i,neg in enumerate(neg_list):
                    rows[f"neg_{i+1}"].append(neg)
    return Dataset.from_dict(rows)

class TripletCollator(DataCollatorWithPadding):
    """
    返回:
      {
        "queries":  {"input_ids": (B,Lq), "attention_mask": (B,Lq)},
        "passages": {"input_ids": (B*(1+N),Ld), "attention_mask": (B*(1+N),Ld)},
        "teacher_scores": None,
        "no_in_batch_neg_flag": False
      }
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        neg_num: int,
        query_max_len: int = 64,
        passage_max_len: int = 256,
        padding: bool | str = True,
        pad_to_multiple_of: int | None = None,
        return_tensors: str = "pt",
        sub_batch_size: int = -1,             # FlagEmbedding 支持切小 sub-batch
    ):
        super().__init__(
            tokenizer=tokenizer,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors
        )
        self.neg_num        = neg_num
        self.query_max_len  = query_max_len
        self.passage_max_len= passage_max_len
        self.sub_batch_size = sub_batch_size  # 若想梯度累积，可以用
        # 上面 4 个字段就是 AbsEmbedderCollator 里的同名配置
        self.valid_label_columns = ("labels",)
        self.return_tensors      = "pt"
        

    # ------------------------------------------------------
    def __call__(self, features):
        """
        features: List[dict]，每个 dict 至少包含
          {
            "query": str,
            "pos_doc": str,
            "neg_1": str, ..., "neg_N": str
          }
        """
        B = len(features)

        # 1) 收集文本
        queries = [f["query"] for f in features]

        passages_flat = []
        for f in features:
            # 先正样本，再 N 个负样本（顺序千万别改）
            passages_flat.append(f["pos_doc"])
            for i in range(self.neg_num):
                passages_flat.append(f[f"neg_{i+1}"])

        # 2) tokenizer（与 FlagEmbedding 方式完全一致）
        q_inputs = self.tokenizer(
            queries, truncation=True, max_length=self.query_max_len,
            return_tensors=None   # 先留 dict，稍后 pad
        )
        p_inputs = self.tokenizer(
            passages_flat, truncation=True, max_length=self.passage_max_len,
            return_tensors=None
        )

        # 3) pad（FlagEmbedding 支持 sub_batch_size，我们也保留）
        def _pad(inputs):
            return self.tokenizer.pad(
                inputs,
                padding=self.padding,
                max_length=self.query_max_len if inputs is q_inputs else self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors
            )

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = _pad(q_inputs)     # dict -> {"input_ids":..., "attention_mask":...}
            p_collated = _pad(p_inputs)
        else:
            # 切小块；和原 AbsEmbedderCollator 一样
            def _chunk_and_pad(inputs):
                sub_feats = []
                bs = self.sub_batch_size
                for i in range(0, len(inputs["attention_mask"]), bs):
                    piece = {k: v[i:i+bs] for k,v in inputs.items()}
                    sub_feats.append(_pad(piece))
                return sub_feats
            q_collated = _chunk_and_pad(q_inputs)
            p_collated = _chunk_and_pad(p_inputs)

        return {
            "queries_input_ids":  q_collated,
            "passages_input_ids": p_collated,
            "teacher_scores": None,       # 你现在没用 teacher
            "no_in_batch_neg_flag": False # flagembedding 的开关，保持 False 表示需要 in-batch neg
        }