from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="relu_scatter_cuda",
    ext_modules=[
        CUDAExtension(
            name="relu_scatter_cuda",
            sources=["relu_scatter.cpp", "relu_scatter_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo", "-use_fast_math"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
