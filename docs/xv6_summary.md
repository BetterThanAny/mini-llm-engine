# xv6 OS Lab 总结

> 面试前 5 分钟讲解稿。`[填]` 处按个人实现补全。

## 一句话简介

MIT 6.S081 xv6-riscv 实验,在 RISC-V xv6 内核上完成系统调用、页表、锁、文件系统等 lab,理解现代 OS 的核心机制。

## 完成的主要 Lab

- [填:勾选/补充]
- [x] Lab util / syscall:添加新系统调用
- [x] Lab pgtbl:页表、vmprint、super page
- [x] Lab traps:backtrace、alarm
- [x] Lab copy-on-write fork
- [x] Lab multithreading:用户态线程切换
- [x] Lab locks:buffer cache / kmem 并发优化
- [x] Lab file system:大文件、符号链接
- [x] Lab mmap
- [x] Lab net

## 架构与关键设计

### 系统调用完整链路

```
用户态
  syscall wrapper (usys.S)
    └─ li a7, SYS_xxx
    └─ ecall                    ← 触发 trap
----------------------- 特权态切换 -----------------------
  trampoline.S:uservec
    └─ 保存用户寄存器到 trapframe
    └─ 切换 satp 到 kernel pagetable
    └─ 跳转到 usertrap()
  trap.c:usertrap()
    └─ 判断 scause:syscall / 时钟中断 / page fault
    └─ 调用 syscall()
  syscall.c:syscall()
    └─ 从 trapframe 取 a7,查 syscalls[] 表
    └─ 执行对应 sys_xxx()
  返回路径:usertrapret → userret → sret 回用户态
```

### 页表结构 (RISC-V Sv39)

- **三级页表**:VPN[2] (9bit) → VPN[1] (9bit) → VPN[0] (9bit) → offset (12bit)
- **PTE 格式**:PPN (44bit) + RSW (2bit) + D/A/G/U/X/W/R/V (8bit)
- **每进程两套页表**:用户态 pagetable + 内核态 pagetable (xv6 每进程独立 kstack)
- **TRAMPOLINE / TRAPFRAME**:映射到每个进程虚拟地址空间顶部,用于 trap 进出

### 并发同步

- **spinlock**:关中断 + CAS 自旋,用于短临界区
- **sleeplock**:可睡眠,持有时可被调度,用于长 IO 操作 (如磁盘读写)
- **xv6 调度核心**:`scheduler() ↔ sched() ↔ swtch()`,per-CPU scheduler context 与进程 context 切换

## 难点 Top 3

1. **[填: 难点 1 — 例如 COW fork 中引用计数与 page fault 处理]**
   - 问题表现:
   - 如何定位:
   - 解决方案:

2. **[填: 难点 2 — 例如 Lab locks 的 buffer cache 哈希桶细粒度锁]**
   - 问题表现:deadlock / 测试卡死
   - 如何定位:
   - 解决方案:按 bucket 分锁,淘汰时按固定顺序取锁避免死锁

3. **[填: 难点 3 — 例如 mmap 懒分配 + page fault 按需读文件]**
   - 问题表现:
   - 如何定位:
   - 解决方案:

## 最难的 Bug

**现象**:[填,例:某个测试偶发 panic / kernel page fault at xxx]
**定位过程**:[填: print scause/sepc、看 backtrace、用 qemu gdb 断点]
**根因**:[填,例: 忘了 flush TLB / 没持有锁访问共享数据 / 释放了还在使用的 page]
**修复**:[填]
**教训**:[填]

## 并发 / 内存相关经典 bug 排查思路

- **死锁**:按锁获取顺序排序、用 `holding()` 断言检查
- **race**:`PRINT` 大法缩小范围,或临时加全局锁看是否还出错
- **内存错误**:检查 `kalloc` / `kfree` 是否匹配,use-after-free,double-free
- **页表**:`vmprint` 看映射是否正确,flag 是否齐全 (U/R/W/X/V)

## 面试高频追问

- **用户态和内核态切换的开销来自哪里?** → 寄存器保存/恢复 + satp 切换 + TLB flush (xv6 每次切满 flush) + cache 污染
- **为什么要有三级页表而不是一级?** → 稀疏地址空间下大幅节省页表内存;二级/三级按需分配
- **spinlock 和 mutex 的区别?** → spinlock 关中断不让出 CPU,mutex 睡眠让出;根据临界区长度选
- **COW fork 的引用计数竞争怎么处理?** → 对 refcnt 加锁 (或用原子操作),fork 时 +1,free 时 -1 到 0 才真正释放
- **xv6 和 Linux 的差距?** → 单 CPU 调度简化、无 SMP 负载均衡、文件系统简化无日志之外的特性、无虚拟文件系统层、调度器简单 round-robin
- **PagedAttention 和 OS 虚拟内存怎么类比?** → KV Cache 的 block 相当于 page,block table 相当于 page table,可以做到 KV 非连续分配、copy-on-write 共享 prompt

## 可以迁移到 AI Infra 的能力

- **虚拟内存 → PagedAttention**:vLLM 核心思想就是把 OS 页表机制搬到 KV Cache 管理;能直接用 xv6 经验讲清楚
- **并发同步 → GPU 同步**:`__syncthreads()` 与 barrier,atomic 操作,warp 级同步
- **系统调用开销 → kernel launch overhead**:用户/内核态切换类比 CPU/GPU 切换,都是 context switch 带来的开销
- **调度 → Continuous Batching**:iteration-level 调度与 OS 进程调度类似,有优先级、抢占、公平性权衡
- **底层 debug 能力**:gdb / qemu / print 调试,在 CUDA 领域用 cuda-gdb / printf / compute-sanitizer 同样适用
