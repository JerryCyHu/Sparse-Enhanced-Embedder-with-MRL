Download bge-large-zh-v1.5 as sentence_transformer. Place it in ./bge-large-zh-v1.5
Install FlashAttansion, Deepspeed, NVCC and requirements_noversion.txt

build the sparse kernal as python module using:
pip install -e .
under cuda_relu_scatter

three options to run:
bge_large_se_mrl_with_everything, which contains the optimized Sparse Retrieval Kernal, FlashAttn and Deepspeed
bge_large_se_mrl_with_flashattn, which contains FlashAttn and Deepspeed
bge_large_se_mrl_mps is for mac with mps architects.