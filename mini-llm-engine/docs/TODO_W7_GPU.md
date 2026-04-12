# W7 GPU 机器待办事项

在 RTX 3080 机器上按顺序完成以下步骤。

---

## 1. 构建

```bash
cd /path/to/mini-llm-engine
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## 2. 正确性验证

### 2a. Flash Attention 算法验证（纯 PyTorch，无需 GPU binary）
```bash
python tests/test_flash_attn.py
# 期望：所有 7 个 case PASS，max_abs_err < 1e-2
```

### 2b. GPU vs CPU token 匹配率
```bash
MODEL_PATH=/path/to/tinyllama-fp32.bin \
TOKENIZER_PATH=/path/to/tokenizer.model \
python tests/test_gpu_cpu_match.py
# 期望：match rate >= 95%
```

---

## 3. 基准测试

### 3a. GPU FP16（Flash Attention W6）
```bash
MODEL_PATH=/path/to/tinyllama-fp32.bin \
TOKENIZER_PATH=/path/to/tokenizer.model \
DEVICE=cuda \
bash benchmarks/run_benchmark.sh
# 结果自动写入 benchmark.csv
```

### 3b. GPU INT8（W8A16）
```bash
./build/llm_engine \
    --model      /path/to/tinyllama-fp32.bin \
    --tokenizer  /path/to/tokenizer.model    \
    --prompt     "The quick brown fox jumps over the lazy dog. In the beginning, there was light, and the light was good. Scientists have long studied the mysteries of the universe, from the smallest subatomic particles to the vast cosmic structures that span billions of light-years across the observable universe." \
    --device cuda --quant int8                \
    --max_new_tokens 128 --temperature 0 --top_k 1 --seed 42 \
    --benchmark --warmup 3 --runs 5           \
    --output_csv benchmark.csv
```

### 3c. Batch throughput（batch=1/2/4/8）
```bash
for BS in 1 2 4 8; do
    ./build/llm_engine \
        --model     /path/to/tinyllama-fp32.bin \
        --tokenizer /path/to/tokenizer.model    \
        --prompt    "The quick brown fox jumps over the lazy dog. In the beginning, there was light, and the light was good. Scientists have long studied the mysteries of the universe, from the smallest subatomic particles to the vast cosmic structures that span billions of light-years across the observable universe." \
        --device cuda --batch_size $BS           \
        --max_new_tokens 128 --temperature 0 --top_k 1 --seed 42 \
        --benchmark --warmup 3 --runs 5          \
        --output_csv benchmark.csv
done
```

---

## 4. 填写 benchmark 数字

把上面跑出来的数字填入：
- `docs/week6.md` 中的 benchmark 表格（GPU FP16 行）
- `docs/week7.md` 中的两个 benchmark 表格（FP16 vs INT8 vs batch）

---

## 5. 收尾

```bash
git add -A
git commit -m "W7: INT8 quant, batch inference, correctness tests, benchmarks"
git tag week7-done
```
