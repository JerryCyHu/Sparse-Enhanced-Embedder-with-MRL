#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void relu_amax_scatter_kernel(
    const scalar_t* __restrict__ hidden,   // [(B*SEQ), H]
    const scalar_t* __restrict__ weight,   // [H]
    const int32_t*  __restrict__ token_id, // [(B*SEQ)]
    scalar_t*       __restrict__ output,   // [B, vocab]
    int B, int SEQ, int H, int vocab)
{
    // 一个 thread 负责一个 token 行
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B * SEQ) return;

    scalar_t acc = 0;
    #pragma unroll 4
    for (int h = 0; h < H; ++h) {
        acc += hidden[row * H + h] * weight[h];
    }

    // ReLU
    acc = acc > 0 ? acc : 0;

    //atomicMax 到 [batch_idx, token_id]
    int batch = row / SEQ;
    int idx   = token_id[row];
    scalar_t* out_ptr = output + batch * vocab + idx;

#if __CUDA_ARCH__ >= 700
    // 直接原子
    atomicMax(out_ptr, acc);
#else
    // 若显卡不支持 atomicMax(float)，用 CAS 循环
    unsigned int* addr_as_ui = (unsigned int*)out_ptr;
    unsigned int old = *addr_as_ui, assumed;
    unsigned int val = __float_as_uint(acc);
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= acc) break;
        old = atomicCAS(addr_as_ui, assumed, val);
    } while (assumed != old);
#endif
}

void launch_relu_scatter(
    at::Tensor hidden, at::Tensor weight,
    at::Tensor token_id, at::Tensor output)
{
    const int B = hidden.size(0);
    const int SEQ = hidden.size(1);
    const int H = hidden.size(2);
    const int vocab = output.size(1);

    auto hidden_2d = hidden.view({B * SEQ, H});

    const int threads = 256;
    const int blocks  = (B * SEQ + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(hidden.scalar_type(), "relu_scatter_cuda", ([&] {
        relu_amax_scatter_kernel<scalar_t><<<blocks, threads, 0,
            at::cuda::getCurrentCUDAStream()>>>(
                hidden_2d.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                token_id.view({B * SEQ}).data_ptr<int32_t>(),
                output.data_ptr<scalar_t>(),
                B, SEQ, H, vocab);
    }));

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "relu_amax_scatter kernel launch failed");
}
