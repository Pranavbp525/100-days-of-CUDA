# 100-days-of-CUDA

100 days of CUDA challenge by github.com/hkproj

## Daily Progress

### Day 1
- Implemented vector addition using CUDA kernel
- Basic CUDA programming concepts: memory allocation, kernel launch, thread indexing

### Day 2
- Developed color to grayscale conversion kernel
- Worked with 2D grid and block dimensions
- Processed image data in CUDA

### Day 3
- Created blur kernel for grayscale images
- Implemented simple image filtering
- Handled edge cases in image processing

### Day 4
- Built naive squared matrix multiplication kernel
- Basic matrix operations in CUDA
- Global memory access patterns

### Day 5
- Implemented matrix-vector multiplication
- Explored different memory access patterns
- Optimized memory coalescing

### Day 6
- Developed tiled matrix multiplication
- Introduced shared memory usage
- Improved performance through memory locality

### Day 7
- Enhanced tiled matrix multiplication
- Added boundary conditions
- Supported non-square matrix multiplication

### Day 8
- Created matrix transpose kernel
- Optimized memory access patterns for transpose
- Explored coarsened tiled matrix multiplication

### Day 9
- Implemented coarsened tiled matrix multiplication
- Increased thread workload
- Improved resource utilization

### Day 10
- Developed basic 2D convolution kernel
- Implemented host-side convolution verification
- Worked with different filter sizes and padding

### Day 11
- Implemented constant memory 2D convolution kernel
- Utilized constant memory for filter coefficients
- Reduced global memory accesses

### Day 12
- Developed tiled 2D convolution kernel
- Used shared memory for input tiles
- Improved memory access patterns

### Day 13
- Created basic 1D convolution kernel
- Implemented simple signal processing
- Worked with 1D memory access patterns

### Day 14
- Built basic 3D convolution kernel
- Extended convolution to volumetric data
- Handled 3D memory access patterns

### Day 15
- Enhanced tiled 2D convolution with cached halo cells
- Optimized shared memory usage
- Improved boundary handling

### Day 16
- Implemented simple 3D stencil kernel
- Basic neighborhood operations in 3D
- Global memory implementation

### Day 17
- Developed tiled 3D stencil kernel
- Used shared memory for 3D tiles
- Improved memory locality

### Day 18
- Enhanced tiled 3D stencil with thread coarsening
- Increased thread workload
- Improved resource utilization

### Day 19
- Created register tiling 3D stencil kernel
- Optimized register usage
- Reduced shared memory pressure

### Day 20
- Implemented basic parallel histogram using atomic add
- Counted character frequencies
- Grouped results into bins

### Day 21
- Developed histogram kernel with private versions in global memory
- Reduced atomic operation conflicts
- Improved performance through privatization

### Day 22
- Created privatized text histogram using shared memory
- Minimized global memory atomic operations
- Utilized shared memory for intermediate results

### Day 23
- Implemented histogram kernel with coarsening (contiguous partitioning)
- Increased thread workload
- Improved memory access patterns

### Day 24
- Developed histogram kernel with coarsening (interleaved partitioning)
- Alternative coarsening approach
- Compared performance with contiguous partitioning

### Day 25
- Built aggregated text histogram kernel
- Combined multiple optimization techniques
- Achieved higher performance histogram

### Day 26
- Implemented simple sum reduction kernel
- Basic parallel reduction pattern
- Global memory implementation

### Day 27
- Developed convergent reduction sum kernel
- Reduced control divergence
- Improved execution resource utilization

### Day 28
- Created shared memory reduction sum kernel
- Minimized global memory accesses
- Optimized memory access patterns

### Day 29
- Implemented segmented multiblock sum reduction
- Used atomic operations for global reduction
- Handled multiple segments in parallel

### Day 30
- Developed sum reduction with thread coarsening
- Increased thread workload
- Improved resource utilization efficiency

### Day 31
- Implemented Kogge-Stone inclusive scan kernel
- Worked with parallel prefix sum algorithm
- Optimized for warp-level operations

### Day 32
- Developed Kogge-Stone exclusive scan variant
- Handled boundary conditions
- Compared performance with inclusive version

### Day 33
- Created Brent-Kung inclusive segmented scan
- Processed multiple segments in parallel
- Handled segment flags and boundaries

### Day 34
- Implemented coarsened inclusive scan kernel
- Increased thread workload
- Improved memory access efficiency

### Day 35
- Developed hierarchical scan using Kogge-Stone
- Combined block-level and global scans
- Optimized multi-level reduction

### Day 36
- Created single pass scan for arbitrary length inputs
- Handled variable-sized inputs efficiently
- Optimized memory access patterns

### Day 37
- Implemented Brent-Kung exclusive scan kernel
- Worked with binary tree reduction pattern
- Optimized shared memory usage

### Day 38
- Developed basic parallel merge sort with co-rank
- Implemented co-rank function for merging
- Worked with sorted sequences

### Day 39
- Created coarsened exclusive scan kernel
- Increased thread workload
- Improved resource utilization

### Day 40
- Implemented Brent-Kung hierarchical scan kernel
- Combined block-level and global scans
- Optimized multi-level reduction pattern

### Day 41
- Developed Brent-Kung single pass scan kernel
- Optimized for single pass execution
- Reduced global memory transactions

### Day 42
- Created tiled merge kernel
- Implemented parallel merging of sorted sequences
- Used shared memory for efficient merging

### Day 43
- Enhanced tiled merge kernel with circular buffer
- Improved memory access patterns
- Reduced shared memory bank conflicts

### Day 44
- Implemented SPMV COO kernel
- Sparse matrix-vector multiplication in COO format
- Handled irregular memory access patterns

### Day 45
- Developed SPMV CSR kernel
- Optimized sparse matrix-vector multiplication
- Utilized CSR format for better memory efficiency

### Day 46
- Created SPMV ELL kernel
- Implemented ELLPACK format sparse matrix multiplication
- Handled padded matrix rows efficiently

### Day 47
- Built SPMV JDS kernel
- Implemented jagged diagonal storage format
- Optimized for matrices with varying row lengths

### Day 48
- Developed BFS vertex-centric push kernel
- Breadth-first search using push-based approach
- Optimized for frontier expansion

### Day 49
- Created BFS vertex-centric pull kernel
- Alternative BFS implementation using pull approach
- Optimized for different graph structures

### Day 50
- Implemented BFS edge-centric kernel
- Breadth-first search processing edges in parallel
- Efficient frontier processing for large graphs

### Day 51
- Implemented vertex-centric push-based BFS using CUDA
- Used CSR (Compressed Sparse Row) graph representation
- Managed frontiers for level-synchronous traversal

### Day 52
- Developed private vertex-centric push-based BFS with shared memory
- Used per-block private frontiers to reduce global memory contention
- Improved parallelism and efficiency for BFS traversal

### Day 53
- Implemented ReLU (Rectified Linear Unit) forward pass kernel
- Compared CPU and GPU performance for large arrays
- Verified correctness and measured speedup

### Day 54
- Developed both forward and backward ReLU kernels
- Compared CPU and GPU implementations for both passes
- Measured performance and verified gradients

### Day 55
- Implemented naive softmax forward kernel for batched data
- Used per-sample normalization for numerical stability
- Verified output sums to 1 for each sample

### Day 56
- Developed optimized softmax forward with multi-threaded kernel
- Used shared memory and parallel reduction for max and sum
- Improved performance for large batch and feature sizes

### Day 57
- Implemented softmax forward with kernel fusion
- Combined exponentiation and sum in a single kernel for efficiency
- Verified output and measured execution time

### Day 58
- Built naive softmax backward kernel for batched data
- Used analytical gradient formula for softmax
- Verified gradients for a few samples

### Day 59
- Implemented optimized softmax backward kernel with reduction
- Used shared memory and parallel reduction for dot product computation
- Improved efficiency for large batch and feature sizes

### Day 60
- Developed basic GEMM (General Matrix-Matrix Multiplication) kernel with shared memory tiling
- Used block tiling and shared memory for efficient matrix multiplication
- Measured performance and verified results

### Day 61
- Implemented optimized GEMM (General Matrix-Matrix Multiplication) kernel with thread coarsening
- Used shared memory tiling and 2x2 output per thread for improved performance
- Included benchmarking and verification in main function

### Day 62
- Developed basic GEMV (General Matrix-Vector Multiplication) kernel
- Assigned one thread per output element for matrix-vector multiplication
- Included correctness verification and performance measurement

### Day 63
- Implemented optimized GEMV kernel with thread coarsening and shared memory
- Each thread processes multiple rows for improved efficiency
- Verified results and measured performance

### Day 64
- Developed max pooling forward pass kernel for 4D tensors
- Used CUDA to compute maximum values and indices for each pooling window
- Prepared indices for use in backward pass

### Day 65
- Implemented average pooling forward pass kernel for 4D tensors
- Computed mean value in each pooling window using CUDA
- Verified output for correctness

### Day 66
- Built max pooling backward pass kernel
- Used indices from forward pass to propagate gradients
- Combined forward and backward max pooling in a single program

### Day 67
- Developed average pooling backward pass kernel
- Evenly distributed gradients to all input elements in each pooling window
- Combined forward and backward average pooling in a single program

### Day 68
- Implemented layer normalization forward pass kernel
- Used shared memory for efficient mean and variance computation per sample
- Applied normalization, scaling, and shifting for each feature in a batch
- Verified output statistics and performance on sample data

### Day 69
- Developed layer normalization backward pass kernels
- Computed gradients for gamma, beta, and input efficiently using shared memory
- Separated parameter and input gradient computations for clarity
- Verified gradient correctness and measured execution time

### Day 70
- Built batch normalization forward pass kernel supporting both training and inference modes
- Used shared memory for per-channel mean and variance computation
- Maintained running statistics for inference
- Verified output and running statistics, and compared training vs inference performance

### Day 71
- Implemented batch normalization backward pass kernel
- Computed gradients for input, gamma, and beta efficiently using shared memory
- Verified correctness and performance for backward propagation

### Day 72
- Developed dropout regularization forward and backward kernels
- Used cuRAND for random mask generation during training
- Compared training and inference behavior, and measured dropout statistics

### Day 73
- Implemented Adam optimizer update kernel
- Supported bias correction for first and second moments
- Benchmarked optimizer on large parameter arrays

### Day 74
- Built cross-entropy loss forward pass kernel
- Computed per-sample loss from softmax predictions and target labels
- Verified loss statistics and correctness on sample data

### Day 75
- Developed cross-entropy loss backward pass kernel
- Calculated gradients with respect to softmax predictions
- Verified gradient correctness and non-zero count for targets

### Day 76
- Implemented neural network layer with cuBLAS-accelerated GEMM
- Added layer normalization and fused softmax + cross-entropy loss
- Benchmarked forward and backward passes for performance

### Day 77
- Built self-attention forward pass using cuBLAS and CUDA kernels
- Projected inputs to Q, K, V and computed scaled dot-product attention
- Applied softmax and aggregated outputs for attention mechanism

### Day 78
- Developed self-attention backward pass with cuBLAS and CUDA
- Implemented gradient computation for softmax and attention scores
- Verified gradients for Q, K, V, and input tensors

### Day 79
- Implemented multi-head attention forward pass with cuBLAS
- Projected inputs to multiple heads and computed parallel attention
- Concatenated head outputs and applied output projection

### Day 80
- Built multi-head attention backward pass kernel
- Computed gradients for all attention parameters and input
- Verified correctness of backward propagation for multi-head attention

### Day 81
- Implemented masked multi-head attention forward pass with cuBLAS
- Applied causal masking to attention scores for autoregressive models
- Verified correctness of masking and output aggregation

### Day 82
- Developed masked multi-head attention backward pass with cuBLAS
- Computed gradients for masked attention, including causal mask handling
- Verified gradients for all attention parameters and input

### Day 83
- Built multi-head cross-attention forward and backward pass with cuBLAS
- Supported separate encoder and decoder inputs for cross-attention
- Verified correctness of gradients for both input streams

### Day 84
- Implemented fused attention forward kernel with shared memory tiling
- Combined softmax and value aggregation in a single CUDA kernel
- Benchmarked performance and verified output

### Day 85
- Developed fused attention backward kernel for efficient gradient computation
- Supported log-sum-exp trick for numerical stability
- Verified gradients for Q, K, V, and output tensors

### Day 86
- Built FlashAttention-style forward pass kernel
- Used tiling and online softmax for memory-efficient attention
- Benchmarked performance on long sequences

### Day 87
- Implemented FlashAttention forward and backward pass kernels
- Supported efficient gradient computation for Q, K, V, and output
- Verified correctness and performance of both passes

### Day 88
- Developed positional encoding generation and application kernels
- Generated sinusoidal positional encodings and added to input embeddings
- Verified correctness of encoding and application

### Day 89
- Implemented rotary positional encoding (RoPE) kernel
- Applied rotary embeddings to query matrices for attention
- Verified correctness and position-dependent transformations

### Day 90
- Built SwiGLU activation forward pass kernel
- Combined Swish and gating mechanism for MLP layers
- Verified output and compared with standard activations

### Day 91
- Developed SwiGLU backward pass kernel for gradient computation
- Calculated gradients for both gated projection and gate values
- Verified correctness of gradients and parameter updates

### Day 92
- Implemented Grouped-Query Attention (GQA) forward pass kernel
- Supported different numbers of query and key-value heads for efficient attention
- Verified correctness of head grouping and output aggregation

### Day 93
- Developed Grouped-Query Attention (GQA) backward pass kernel
- Computed gradients for all GQA parameters and input
- Verified correctness of gradients for grouped attention

### Day 94
- Built 2D convolution forward pass using cuDNN
- Set up tensor and filter descriptors for cuDNN API
- Verified convolution output and cuDNN algorithm selection

### Day 95
- Implemented a full CNN layer with cuDNN: convolution, batch normalization, ReLU, and max pooling
- Chained multiple cuDNN operations for a typical CNN block
- Verified output at each stage of the pipeline

### Day 96
- Developed INT8 quantization and dequantization kernels for matrix multiplication
- Implemented custom INT8 GEMM and compared with FP32 reference
- Measured quantization error and verified output correctness

### Day 97
- Built NF4 quantization and dequantization kernels for weights
- Quantized weights to 4-bit NF4 format and reconstructed them on GPU
- Measured quantization error and verified output

### Day 98
- Implemented QLoRA forward and backward pass kernels
- Combined NF4 quantized base weights with LoRA adapters for efficient fine-tuning
- Verified gradients for LoRA parameters and output correctness

### Day 99
- Developed Mixture-of-Experts (MoE) step with top-1 routing
- Implemented token dispatch, expert computation, and output aggregation on GPU
- Verified expert assignment and output correctness

### Day 100
- Built denoising diffusion sampling step kernel
- Implemented reverse diffusion process for image generation
- Verified denoising steps and final image output
