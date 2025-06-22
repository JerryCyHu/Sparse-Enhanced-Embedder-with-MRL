import torch
import importlib
from pathlib import Path

_src = Path(__file__).parent
relu_scatter_cuda = importlib.import_module(
    torch.utils.cpp_extension.load(
        name="relu_scatter_cuda",
        build_directory=str(_src / "build"),
        sources=[str(_src / "relu_scatter.cpp"),
                 str(_src / "relu_scatter_kernel.cu")],
        verbose=False))

def relu_amax_scatter(hidden, token_id, weight, vocab_size):
    """
    hidden:  [B, SEQ, H]      float32/float16
    token_id:[B, SEQ]          int64/int32
    weight:  [H]               same dtype as hidden
    """
    out = torch.zeros(
        (hidden.size(0), vocab_size),
        dtype=hidden.dtype, device=hidden.device)
    relu_scatter_cuda.forward(hidden, weight, token_id.to(torch.int32), out)
    return out
