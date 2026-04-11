## CUDA 编程模型

### 核心思想

CPU 擅长复杂逻辑，GPU 擅长大量简单并行计算。CUDA 让你写一个函数（**kernel**），让 GPU 上成千上万个线程同时执行它。

### 1. Kernel 函数与 `<<<grid, block>>>` 启动

```cuda
// __global__ 标记这是一个 GPU 上执行、CPU 上调用的函数
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];  // 每个线程只算一个元素
    }
}

// CPU 端启动 kernel
vector_add<<<grid, block>>>(a, b, c, n);
//         ^^^grid^^^block^^
//         多少个 block | 每个 block 多少个 thread
```

三种函数修饰符：

| 修饰符 | 在哪执行 | 谁调用 |
|--------|---------|--------|
| `__global__` | GPU | CPU（或 GPU，动态并行） |
| `__device__` | GPU | GPU |
| `__host__` | CPU | CPU（默认，可省略） |

### 2. Thread / Block / Grid 层级

```
Grid（网格）
├── Block (0,0)          Block (1,0)          Block (2,0)
│   ├── Thread 0         ├── Thread 0         ├── Thread 0
│   ├── Thread 1         ├── Thread 1         ├── Thread 1
│   ├── ...              ├── ...              ├── ...
│   └── Thread 255       └── Thread 255       └── Thread 255
```

**关键变量**：

| 变量 | 含义 |
|------|------|
| `threadIdx.x` | 当前线程在 block 内的编号 |
| `blockIdx.x` | 当前 block 在 grid 内的编号 |
| `blockDim.x` | 每个 block 有多少线程 |
| `gridDim.x` | grid 有多少个 block |

**全局索引计算**（最常用的一行代码）：

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

举例：`blockDim.x = 256`，`blockIdx.x = 3`，`threadIdx.x = 10`

```
i = 3 * 256 + 10 = 778  → 这个线程负责第 778 号元素
```

**启动配置怎么算**：

```cuda
int n = 1000000;
int block_size = 256;  // 每个 block 256 个线程（常用值）
int grid_size = (n + block_size - 1) / block_size;  // 向上取整
kernel<<<grid_size, block_size>>>(...);
```

### 3. GPU 内存管理

CPU 和 GPU 有各自独立的内存，数据需要显式搬运：

```
CPU (Host)                    GPU (Device)
┌──────────┐    cudaMemcpy    ┌──────────┐
│  float*  │ ──────────────→  │  float*  │
│  h_a     │                  │  d_a     │
└──────────┘  ←──────────────  └──────────┘
   malloc()     cudaMemcpy      cudaMalloc()
   free()                       cudaFree()
```

```cuda
float *h_a, *d_a;           // h_ = host, d_ = device（命名惯例）
int size = n * sizeof(float);

// CPU 端分配
h_a = (float*)malloc(size);

// GPU 端分配
cudaMalloc(&d_a, size);

// 数据搬运：CPU → GPU
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

// 运行 kernel...

// 数据搬运：GPU → CPU
cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

// 释放
cudaFree(d_a);
free(h_a);
```

### 4. nvcc 编译基本用法

```bash
# 基本编译
nvcc vector_add.cu -o vector_add

# 常用选项
nvcc -O2 vector_add.cu -o vector_add        # 开优化
nvcc -arch=sm_86 vector_add.cu -o vector_add # 指定 GPU 架构（3080 = sm_86）
nvcc -G vector_add.cu -o vector_add          # 开 debug 信息
```

`nvcc` 会把代码分成两部分：CPU 代码交给 `gcc/g++`，GPU 代码自己编译成 PTX → SASS。
