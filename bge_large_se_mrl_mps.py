import os
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MatryoshkaLoss
from bge_large_se_mrl_helper import *
from transformers import TrainerCallback, TrainerState, TrainerControl
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────── Dense + Sparse wrapper ────────────────────────

def main():

    train_ds = load_triplets("training.json")
    train_ds = train_ds.shuffle(seed=42)
    # ───────────────────────────── Train Script ─────────────────────────────

    dense_base = SentenceTransformer("./bge-large-zh-v1.5")
    model = DenseSparseModel(dense_base).to(dense_base.device)
    lfn = CustomInfoNCELoss(model, temperature=0.02)
    loss_fn = MatryoshkaLoss(model=model, loss=lfn, matryoshka_dims=[1024, 512, 256])
    model.save_pretrained("bge_finetuned_se_mrl/final", safe_serialization=False)

    args = SentenceTransformerTrainingArguments(
        output_dir="bge_finetuned_se_mrl",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # adjust to your GPU/CPU memory
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=False,                     # set to True if you have a compatible GPU
        bf16=False,
        logging_steps=20000,
        save_strategy="epoch",
        gradient_accumulation_steps=1, 
        save_safetensors=False,
        #fp16_backend="auto",
    )
    tokenizer = dense_base.tokenizer
    collator  = TripletCollator(tokenizer, neg_num=NEG_NUM)
    
    class GradInspectCallback(TrainerCallback):
        def __init__(self, param_names=("sparse_linear.weight",)):
            self.watch = set(param_names)

        # 比 on_step_end 早，被调用时梯度尚在
        def on_pre_optimizer_step(
            self, args, state, control, optimizer, **kwargs
        ):
            model = kwargs["model"]
            for n, p in model.named_parameters():
                if n in self.watch:
                    if p.grad is None:
                        print(f"[step {state.global_step}] {n:<30} ⛔️ grad=None")
                    else:
                        g = p.grad
                        print(
                            f"[step {state.global_step}] {n:<30} μ={g.mean():.3e}  "
                            f"max={g.abs().max():.3e}"
                        )
            return control
    class VerboseTrainer(SentenceTransformerTrainer):
        def compute_loss(self, *wargs, **kwargs):
            loss = super().compute_loss(*wargs, **kwargs)
            # 这里直接打印；不要忘了 .item()
            print(f"[step {self.state.global_step}] total loss = {loss.item():.4f}")
            return loss
    trainer = VerboseTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        loss=loss_fn,
        callbacks=[GradInspectCallback(param_names=["sparse_linear.weight", "0.auto_model.embeddings.word_embeddings.weight"])],
    )

    trainer.train()

    # ───────────────────────────── Save Model ─────────────────────────────
    model.save_pretrained("bge_finetuned_se_mrl/final", safe_serialization=False)

if __name__ == "__main__":
    main()