# cuda-hello 编译与执行报告

- 日期：2026-11-05
- 项目：`/home/yu/Documents/gpu/remotecuda/examples/cuda-hello`
- 总体结论：**编译通过；本机执行失败，原因是当前环境没有可用的 CUDA GPU 设备**

## 1. 结果摘要

| 阶段 | 结果 | 说明 |
| --- | --- | --- |
| CUDA 编译 | ✅ 通过 | 生成 `cuda_hello`，目标架构 `sm_86` |
| 链接 | ✅ 通过 | 生成可执行 ELF 文件 |
| GPU 设备检测 | ❌ 失败 | `nvidia-smi` 无法与 NVIDIA 驱动通信 |
| 程序执行 | ❌ 失败 | `cudaMalloc` 返回 `no CUDA-capable device is detected` |
| 功能验证 | ⏸ 未完成 | 需要在有可用 NVIDIA GPU 的机器上重跑 |

## 2. 编译信息

本机未找到 `cmake` 可执行文件，因此本次使用 CUDA 工具链中的 `nvcc` 直接编译测试源码。

```bash
/usr/local/cuda/bin/nvcc \
  -std=c++17 \
  -O3 \
  -arch=sm_86 \
  -Xcompiler -Wall,-Wextra,-Werror \
  main.cu \
  -o build/local/cuda_hello
```

编译结果：

- 退出码：`0`
- 耗时：`495 ms`
- 警告：`0`
- 错误：`0`

产物：

```text
examples/cuda-hello/build/local/cuda_hello
```

- 文件大小：`1,082,856 bytes`
- 文件类型：x86-64 ELF 可执行文件
- SHA256：`31cc4d6012f2b4d794c638376316f70439b37e7523f232902b1df1ff719cb163`
- 源码 SHA256：`35123b7539666d764a1b2b0ea82795ff67c23599635b196ab55192e629a84ecb`

`cuobjdump` 确认二进制中包含 `sm_86` CUDA 代码：

```text
ELF file    1: cuda_hello.1.sm_86.cubin
ELF file    2: cuda_hello.2.sm_86.cubin
```

## 3. 执行信息

执行命令：

```bash
CUDA_VISIBLE_DEVICES=0 ./build/local/cuda_hello
```

实际输出：

```text
cudaMalloc failed: no CUDA-capable device is detected
```

执行结果：

- 退出码：`1`
- 耗时：`9 ms`
- 阻塞点：第一次 CUDA 运行时显存分配 `cudaMalloc`

该测试程序成功执行时应输出：

```text
GPU result: 42
```

并以退出码 `0` 结束。本次没有到达内核执行和结果拷贝阶段。

## 4. 环境信息

| 项目 | 值 |
| --- | --- |
| 操作系统 | Linux x86_64，内核 6.12.107+deb13-amd64 |
| CUDA 编译器 | CUDA 13.3 / nvcc 13.3.73 |
| 主机编译器 | g++ 14.2.0 |
| 目标 GPU 架构 | `sm_86` |
| `nvidia-smi` | 失败：无法与 NVIDIA 驱动通信 |
| GPU 设备节点 | 未检测到可用的 `/dev/nvidia*` 设备 |

`nvidia-smi` 原始错误：

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
Make sure that the latest NVIDIA driver is installed and running.
```

## 5. 结论与后续动作

1. `cuda-hello` 源码和 CUDA 编译链路正常，生成的二进制结构符合预期。
2. 当前机器没有暴露可用的 NVIDIA GPU 设备，因此不能完成运行时功能验证。
3. 需要在以下任一环境中重跑：
   - 有可用 NVIDIA GPU 和驱动的物理机；
   - 已配置 GPU 设备直通的容器；
   - 已启动的 Remote CUDA Agent 远程 GPU 服务器。
4. 远程重跑时，建议将 `examples/cuda-hello` 注册为 Workspace，使用 CMake 构建并执行 `cuda_hello`。

## 6. 证据文件

- 编译日志：`examples/cuda-hello/build-local.log`
- 执行日志：`examples/cuda-hello/run-local.log`
- 可执行文件：`examples/cuda-hello/build/local/cuda_hello`
