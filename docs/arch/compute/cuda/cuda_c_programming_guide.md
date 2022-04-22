#CUDA C Programming Guide
[toc]
---
## 1. Introduction
## 2. Programming Model
### 2.1 Kernels
### 2.2 Thread Hierarchy
- Thread Block
    a thread block may contain up to 1024 threads
    Threads within a block can cooperate by sharing data through some shared memory and by synchronizing their execution to coordinate memory accesses
### 2.3 Memory Hierarchy
### 2.4 Heterogeneous Programming
### 2.5 Asynchronous SIMT programming Model
### 2.6 Compute Capability

---

## 3. Programming Interface
### 3.1
### 3.2 CUDA Runtime
#### 3.2.3 Device Memory L2 Access Management
*persisting* : CUDA kernel accesses a data region in the global memory repeatedly.
*streaming* : the data is only accessed once.
##### 3.2.3.1  L2 Cache Set Aside for persisting accesses
```c++
cudaGetDeviceProperties(&prop, device_id);                
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/ 
```

##### 3.2.3.2 L2 Policy for persisting Accesses
- CUDA Stream Example:


#### 3.2.6 Asynchronous Concurrent Execution
##### 3.2.6.5 Streams
- Applications manage the concurrent operations described above through streams
- A stream is a sequence of commands that execute in order *a stream is asn actual execution queue*
- Different streams, on the other hand, may execute their commands out of order with respect to one another or concurrently

##### 3.2.6.6 Graphs
- A graph is a series of operations, such as kernel launches, connected by dependencies, which is defined separately from its execution *this is similar to OpenGL display lists*
-   Separating out the definition of a graph from its execution enables a number of optimizations:
    - first, CPU launch costs are reduced compared to streams, because much of the setup is done in advance; 
    - second, presenting the whole workflow to CUDA enables optimizations which might not be possible with the piecewise work submission mechanism of streams

- Work submission using graphs is separated into three distinct stages:
    - **definition** During the definition phase, a program creates a description of the operations in the graph along with the dependencies between them.
    - **Instantiation** takes a snapshot of the graph template, validates it, and performs much of the setup and initialization of work with the aim of minimizing what needs to be done at launch. The resulting instance is known as an *executable graph*.
    - **execution** *An executable graph may be launched into a stream*, similar to any other CUDA work. It may be launched any number of times without repeating the instantiation.

**imperative** stream programming model
**declarative** graph based work model


###### Goals
1. Reduce launch overhead for kernels that are very short running (1-10us)
2. Allow developers to express control flow of programs in a "define-once-run-repeatedly" execution flow.
3. Communicate the high-level program structure to CUDA so that the runtime may find optimization opportunities.

###### CUDA Graph Stream Capture
 a bridge from stream based work description to graph base description.

---

## 4. Hardware Implementation
### 4.1 SIMT Architecture
The SIMT architecture is akin to SIMD vector organizations in that a single instruction controls multiple processing elements. 
key difference is that SIMD vector organizations expose the SIMD width to the software, whereas SIMT instructions specify the execution and branching behavior of a single thread. In contrast with SIMD vector machines, SIMT enables programmers to write thread-level parallel code for independent, scalar threads, as well as data-parallel code for coordinated threads.  ***Vector architectures***, on the other hand, require the software to coalesce loads into vectors and manage divergence manually.

- **Independent Thread Scheduling** allows full concurrency between threads, regardless of warp With Independent Thread Scheduling, the GPU maintains execution state per thread, including a program counter and call stack, and can yield execution at a per-thread granularity, either to make better use of execution resources or to allow one thread to wait for data to be produced by another. ***threads can now diverge and reconverge at sub-warp granularity.***
- *The term ***warp-synchronous*** refers to code that implicitly assumes threads in the same warp are synchronized at every instruction.*

**Note**: If a non-atomic instruction executed by a warp writes to the same location in global or shared memory for more than one of the threads of the warp, the number of serialized writes that occur to that location varies depending on the compute capability of the device, and which thread performs the final write is undefined.

If an atomic instruction executed by a warp reads, modifies, and writes to the same location in global memory for more than one of the threads of the warp, each read/modify/write to that location occurs and they are all serialized, but the order in which they occur is undefined.

### 4.2 Hardware Multithreading
- execution context maintained on-chip in warp lifetime, and no cost for context switching. a parallel data cache or shared memory that is partitioned among the thread blocks

- block size determination and occupancy calculation
<br>
---

## 5. Performance Guidelines
### 5.1 Overall Perf Optimization strategies
- Maximize parallel execution to achieve maximum utilization;
- Optimize memory usage to achieve maximum memory throughput;
- Optimize instruction usage to achieve maximum instruction throughput;
- Minimize memory thrashing.

### 5.2 Maximize Utilization

- **Application Level**
    using asynchronous functions calls
    streams (Asynchronous concurrent execution)
    serial workloads to the host; parallel workloads to the devices.

    - For the parallel workloads, at points in the algorithm where parallelism is broken because some threads need to synchronize in order to share data with each other
        - Either these threads belong to the same block, in which case they should use `__syncthreads()` and share data through shared memory within the same kernel invocation
        - they belong to different blocks, in which case they must share data through global memory using two separate kernel invocations, one for writing to and one for reading from global memory. ***less Optimal, adds overhead of extra kernel invocations and global memory traffic***
    <br>

- **Device Level**
    Multiple kernels execute concurrently on a device.
    *Asynchronous concurrent Execution*
    <br>

- **Multiprocessor Level**
    Instruction *latency*; the number of instructions required to hide a latency of **L** clock cycles depends on the respective throughputs of these instructions.
    -   4**L**: a multiprocessor issues one instruction per warp over one clock cycle for four warps(CC5.X,6.1,6.2,7.X,8.X)
    -   2**L**: the two instructions issued every cycle are one instruction for two different warps(CC6.0)
    -   8**L**: the eight instructions issued every cycle are four pairs for four different warps, each pair being for the same warp. (CC3.X)
    <br>
    1. The most common reason a warp is not ready to execute its next instruction is that the **instruction's input operands are not available** yet.
    2. Another reason a warp is not ready to execute its next **instruction is that it is waiting at some memory fence or synchronization point**
    A synchronization point can force the multiprocessor to idle as more and more warps wait for other warps in the same block to complete execution of instructions prior to the synchronization point. Having multiple resident blocks per multiprocessor can help reduce idling in this case, as warps from different blocks do not need to wait for each other at synchronization points.

    - total amount of shared memory required for a block = the amount of statically allocated shared memory + the amount of dynamically allocated shared memory

    - the compiler attempts to minimize register usage while keeping register spilling (device memory access)
    - Occupancy Calculator 
    ```cpp
        // Device code
        __global__ void MyKernel(int *array, int arrayCount)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < arrayCount) {
                array[idx] *= array[idx];
            }
        }
        // Host code
        int launchMyKernel(int *array, int arrayCount)
        {
            int blockSize; // The launch configurator returned block size
            int minGridSize; // The minimum grid size needed to achieve the
            // maximum occupancy for a full device launch
            cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                               &blockSize,
                                               (void*)MyKernel,
                                               0,
                                               arrayCount);
            int gridSize; // The actual grid size needed, based on input size
            gridSize = (arrayCount + blockSize - 1) / blockSize; // Round up according to array size
            MyKernel<<<gridSize, blockSize>>>(array, arrayCount);
            cudaDeviceSynchronize();
            // If interested, the occupancy can be calculated with
            // cudaOccupancyMaxActiveBlocksPerMultiprocessor
            return 0;
        }
    ```
---

### 5.3 Maximize memory throughput 
- **Data Transfer between Host and Device** The first step in maximizing overall memory throughput for the application is to minimize data transfers with low bandwidth(between global memory and device), by maximizing use of on-chip memory: shared memory and caches.
    - move more code from the host to the device
    - batching many small transfers into a single large transfer always performs better than making each transfer separately
    - using page-locked host memory, For maximum performance, these memory accesses must be coalesced as with accesses to global memory, especially for integrated systems (where device memory and host memory are physically the same)

- **Device Memory Access** The next step in maximizing memory throughput is therefore to organize memory accesses as optimally as possible based on the optimal memory access patterns

    - ***Global Memory***
    accessed via 32-, 64-, or 128-byte memory transactions. These memory transactions must be naturally aligned. In general, the more transactions are necessary, the more unused words are transferred in addition to the words accessed by the threads, reducing the instruction throughput accordingly. 
    To maximize global memory throughput, it is therefore important to **maximize coalescing** by:
        - Following the most optimal access patterns based on Compute Capability ?

        - Using data types that meet the size and alignment requirement.
            instructions support reading or writing words of size equal to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to data residing in global memory compiles to a single global memory instruction **if and only if the size of the data type is 1, 2, 4, 8, or 16 bytes** and **the data is naturally aligned**
            >e.g.: `int A[4];` can't generate LDG.128, but `int4 A;` can
            **recommended to use types that meet this requirement for data that resides in global memory.**

            <br>
            For structures, the size and alignment requirements can be enforced by the compiler using the alignment specifiers `__align__(8)` or `__align__(16)`   

            ```cpp
            struct __align__(8) {
                float x;
                float y;
            };   
            ```   
            Any address of a variable residing in global memory or returned by one of the memory allocation routines from the driver or runtime API is always aligned to at least 256 bytes.
            Reading non-naturally aligned 8-byte or 16-byte words produces incorrect results so special care must be taken to maintain alignment of the starting address of any value or array of values of these types (`cudaMalloc()/cuMemAlloc()` is the easy way to overlook this)

            <br>

        - Padding data in some cases.
        a common global memory access pattern:`BaseAddress + width * ty + tx` (use thread of index "tx, ty") to access one element of 2D array (data type meets the requirements described in [maximize utilization](###-5.2-maximize-utilization)
        *For these accesses to be fully coalesced, both the width of the thread block and the width of the array must be a multiple of the warp size.*
        `cudaMallocPitch()` and `cuMemAllocPitch()` and associated memory copy function enable programmers to write non-hardware-dependent code to allocate arrays that conform to these constraints

    - ***Local Memory***
        Automatic variables that the compiler is likely to place in local memory are:
        - Arrays for which it cannot determine that they are indexed with constant quantities
        - Large structures or arrays that would consume too much register space
        - **Any variable if the kernel uses more registers than available** (this is also known as register spilling).

        `ld.local` / `st.local`
        some mathematical functions have implementation paths that might access local memory.    

        The local memory space resides in device memory, so local memory accesses have the same high latency and low bandwidth as global memory accesses and are subject to the same requirements for memory coalescing
        <br>
        Local memory is however organized such that consecutive 32-bit words are accessed by consecutive thread IDs. Accesses are therefore fully coalesced as long as all threads in a warp access the same relative address
        <br>
        local memory accesses are always cached in L1 and L2 in the same way as global memory accesses (CC3.X) and are always cached in L2 in the same way as global memory accesses (CC5.X and 6.X)

    - ***Shared Memory***
    if two addresses of a memory request fall in the same memory bank, there is a bank conflict and the access has to be serialized decreasing throughput by a factor equal to the number of separate memory requests. If the number of separate memory requests is `n`, the initial memory request is said to cause `n-way` bank conflicts

               CCX.X to view details.

    - ***constant Memory***
    The constant memory space resides in device memory and is cached in the constant cache

    - ***Texture and Surface Memory***
    The texture and surface memory spaces reside in device memory and are cached in texture cache, 
        - The texture cache is optimized for 2D spatial locality, so threads of the same warp that read texture or surface addresses that are close together in 2D will achieve best performance
        - it is designed for streaming fetches with a constant latency;

        Reading device memory through texture or surface fetching present some benefits that can make it an advantageous alternative to reading device memory from global or constant memory
        - If the memory reads do not follow the access patterns that global or constant memory reads must follow to get good performance, higher bandwidth can be achieved providing that there is locality in the texture fetches or surface reads;
        - Addressing calculations are performed outside the kernel by dedicated units;
        - Packed data may be broadcast to separate variables in a single operation;
        - 8-bit and 16-bit integer input data may be optionally converted to 32 bit floating-point values in the range [0.0, 1.0] or [-1.0, 1.0] (see Texture Memory).

---
## 5.4 Maximize instruction throughput  
- Minimize the use of arithmetic instructions with low throughput
    - trading precision for speed when it does not affect the end result
- Minimize divergent warps caused by control flow instructions
- Reduce the number of instructions
    - by optimizing out synchronization points whenever possible
    - **by using restricted pointers** ??
    <br>

- **Arithmetic Instructions**
    **Throughput of Native Arithmetic Instructions**(Number of Results per Clock Cycle per Multiprocessor)
    *throughputs are given in number of operations per clock cycle per multiprocessor. For a warp size of 32, one instruction corresponds to 32 operations, so if N is the number of operations per clock cycle, the instruction throughput is N/32 instructions per clock cycle.*

    |Compute Capability |3.5/3.7|5.0/5.2|5.3|6.0|6.1|6.2|7.x|8.0|8.6|
    |:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
    |16-bit hadd, hmul, hmad|n/a|n/a|256|128|2|256|128|256|
    |32-bit fadd, fmul, fmad|192|128|128|64|128|128|64|64|128|
    |64-bit dadd, dmul, dmad|64|4|4|32|4|4|32|32|2|
    |32-bit rcp,rcpsqrt,log2f,exp2f,sinf,cosf|32|32|32|16|32|32|16|16|16|
    |32-bit iadd,extpadd,sub,extpsub|160|128|128|64|128|128|64|64|64|


- **Control Flow instruction**
Any flow control instruction (if, switch, do, for, while) can significantly impact the effective instruction throughput by causing threads of the same warp to diverge
    [see cuda_c_best_practice_guide]


- **synchronization instruction**
Throughput for `__syncthreads()` is 
    128 operations per clock cycle of CC3.X
    32 operations per clock cycle of CC6.0
    16 operations per clock cycle for CC7.X 8.X
    64 operations per clock cycle of CC5.X 6.1 6.2

## 5.5 Minimize memory thrashing
Applications that constantly allocate and free memory too often may find that the allocation calls tend to get slower over time up to a limit. This is typically expected due to the nature of releasing memory back to the operating system for its own use
    1. Try to size your allocation to the problem at hand. Don't try to allocate all available memory with `cudaMalloc / cudaMallocHost / cuMemCreate`, as this forces memory to be resident immediately and prevents other applications from being able to use that memory. This can put more pressure on operating system schedulers, or just prevent other applications using the same GPU from running entirely
    2. Try to allocate memory in appropriately sized allocations early in the application and allocations only when the application does not have any use for it. Reduce the number of `cudaMalloc+cudaFree` calls in the application, especially in performance-critical regions.
    3. If an application cannot allocate enough device memory, consider falling back on other memory types such as `cudaMallocHost` or `cudaMallocManaged`, which may not be as performant, but will enable the application to make progress.
    4. For platforms that support the feature, `cudaMallocManaged` allows for oversubscription, and with the correct `cudaMemAdvise` policies enabled, will allow the application to retain most if not all the performance of `cudaMalloc`. `cudaMallocManaged` also won't force an allocation to be resident until it is needed or prefetched, reducing the overall pressure on the operating system schedulers and better enabling multi-tenet use cases.
    
---
## B. C++ EXTENSIONS
### B.5 Memory Fence Functions
The memory fence functions differ in the scope in which the orderings are enforced but they are independent of the accessed memory space
```cpp
void __threadfence_block();
```

```cpp
void __threadfence();
```

```cpp
void __threadfence_system();
```

## APPENDIX N UNIFIED MEMORY PROGRAMMING.