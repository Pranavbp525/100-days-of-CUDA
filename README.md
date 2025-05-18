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
