Download bge-large-zh-v1.5 as sentence_transformer. Place it in ./bge-large-zh-v1.5
Install FlashAttansion, Deepspeed, NVCC and requirements_noversion.txt

build the sparse kernal as python module using:
pip install -e .
under cuda_relu_scatter

three options to run:
bge_large_se_mrl_with_everything, which contains the optimized Sparse Retrieval Kernal, FlashAttn and Deepspeed
bge_large_se_mrl_with_flashattn, which contains FlashAttn and Deepspeed
bge_large_se_mrl_mps is for mac with mps architects.


I developed the Sparse-Enhanced Embedder with MRL (https://github.com/JerryCyHu/Sparse-Enhanced-Embedder-with-MRL) to tackle the bi-encoder’s difficulty in retrieving documents where key information is sparsely distributed. Starting from BGE-large-zh-v1.5, I integrated Matryoshka Representation Learning (MRL) (arXiv:2205.13147) and the sparse retrieval head concept from BGE-M3 (arXiv:2402.03216). By training the model to allocate attention at the token level—rather than relying solely on the CLS token—it learns to surface fine-grained relevance signals and dramatically improves recall on information-sparse texts.

On the performance side, I originally accelerated the sparse-head kernel with Triton for rapid prototyping. When a C++/CUDA implementation was requested, I ported those kernels with zero loss in throughput. To maximize efficiency, I fused the linear projection, ReLU activation, and scatter aggregation into one custom GPU operator—yielding a 3–4× speedup compared to a naïve sequential approach. This combination of representational enhancements and low-level kernel optimization makes the embedder both more accurate on sparse content and highly efficient in deployment.
