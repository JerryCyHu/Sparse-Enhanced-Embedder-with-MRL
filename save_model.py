from sentence_transformers import SentenceTransformer

model_name     = "BAAI/bge-large-zh-v1.5"
save_directory = "bge-large-zh-v1.5"   # 同级目录

print(f"正在加载 SentenceTransformer：{model_name}")
model = SentenceTransformer(model_name)      # 自动下载 tokenizer + 1_Pooling + 2_Dense

print(f"保存到本地：{save_directory}")
model.save(save_directory)
print("模型与全部组件已保存完毕。")
