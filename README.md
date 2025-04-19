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
