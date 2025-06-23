import os
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MatryoshkaLoss

torch.backends.cuda.enable_flash_sdp(True)
from bge_large_se_mrl_helper import *
from transformers import TrainerCallback, TrainerState, TrainerControl
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────── Dense + Sparse wrapper ────────────────────────
class CheckSparseLinearCallback(TrainerCallback):
    def __init__(self, model):
        # grab the initial weights
        self.prev = model.sparse_linear.weight.detach().cpu().clone()

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # 获取当前权重
        curr = kwargs["model"].sparse_linear.weight.detach().cpu()
        # 计算差值矩阵
        diff = curr - self.prev
        abs_diff = diff.abs()
        # 找到最大改动值和其扁平索引
        max_val = abs_diff.max().item()
        idx = abs_diff.argmax().item()
        # 转成 (row, col)
        n_rows, n_cols = abs_diff.shape
        row = idx // n_cols
        col = idx % n_cols
        print(f"[step {state.global_step}] sparse_linear.weight 最大改动 = {max_val:.3e} at ({row}, {col})")
        # 更新 snapshot
        self.prev = curr.clone()
        return control


def main():

    train_ds = load_triplets("training.json")
    train_ds = train_ds.shuffle(seed=42)
    # ───────────────────────────── Train Script ─────────────────────────────

    dense_base = SentenceTransformer("./bge-large-zh-v1.5")
    dense_base[0].auto_model.config.attention_implementation = "flash_attention_2"  # HF ≥4.38
    dense_base = dense_base.half()
    model = DenseSparseModel(dense_base).to(dense_base.device)
    lfn = CustomInfoNCELoss(model, temperature=0.02)
    loss_fn = MatryoshkaLoss(model=model, loss=lfn, matryoshka_dims=[1024, 512, 256, 128, 64])
    model.save_pretrained("bge_finetuned_se_mrl/final", safe_serialization=False)
    args = SentenceTransformerTrainingArguments(
        output_dir="bge_finetuned_se_mrl",
        num_train_epochs=3,
        per_device_train_batch_size=8,  # adjust to your GPU/CPU memory
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,                     # set to True if you have a compatible GPU
        bf16=False,
        logging_steps=20000,
        save_strategy="epoch",
        gradient_accumulation_steps=2, 
        deepspeed="ds_config.json",
        dataloader_num_workers=16,
        save_safetensors=False,
        #fp16_backend="auto",
    )

    # ▲ Notice: we do NOT pass data_collator here, so Trainer applies its default tokenization
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=loss_fn,
    )

    trainer.train()

    # ───────────────────────────── Save Model ─────────────────────────────
    model.save_pretrained("bge_finetuned_se_mrl/final", safe_serialization=False)

if __name__ == "__main__":
    main()