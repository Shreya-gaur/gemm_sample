## General Matrix Multiply (GEMM)

# Motivation
General Matrix Multiplications (GEMMs) are indeed a fundamental operation in many deep learning workloads and optimizing them is crucial for improving the overall performance of deep learning models. GEMMs involve multiplying two matrices and can be applied to various operations within neural networks, including fully connected layers and convolutional layers when they are converted into matrix multiplications.

# Optimizations
The repository attempts the following optimizations in sample.cu as of now:
1.	CPU
a.	Outer loop first in matrix multiplication
b.	Tiling the output and the inputs
2.	GPU
a.	Corner Turning
b.	Use of Shared Memory
This code at this stage should be considered just as a jump-off point rather than the destination. The limitations involve:
1.	Use of set TILE_WIDTH of 16. Smaller inputs suffer due to this.
2.	A lot of constants like offsets that can be computed in the compile time are computed in runtime.
3.	Tensor Cores are not used.
4.	No kernel fusion is attempted.
5.	Code can be made more readable still.

# Making it Work!
Just run a `make` to generate an executable called `sample`. You can then select which optimization to use once you run the executable. An executable can be created in a debug mode if you run `make debug`.
>	Please note that the compilation is happening for GPU architecture with SM75 (Turing). You should look into Makefile if you want to change that.
