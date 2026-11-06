# Remote CUDA MCP

一个本地 MCP Gateway 管理多台远程 GPU 服务器的 CUDA 开发系统。Coding Agent 只连接本地 MCP Server，即可完成服务器注册、GPU 查询、Workspace 创建、SHA256 增量同步、远程 CMake 编译、CUDA 程序执行、Benchmark、日志查询和产物下载。

## 架构

```text
Codex / OpenCode
      MCP stdio
      v
Local MCP Gateway
  - Server / Workspace / Job Registry (SQLite)
  - Remote Client Pool (httpx)
  - Health Monitor + Resource Scheduler
      HTTP API
      v
Remote CUDA Agent 1..N
  - FastAPI
  - Workspace / Sync / GPU / Build / Job / Artifact Manager
  - CMake + NVCC / nvidia-smi
```

职责边界：

- Local Gateway 负责配置、跨服务器索引、调度和下载编排。
- Remote Agent 只管理本机 Workspace、GPU 锁、任务执行和产物。
- Remote Agent 之间不直接调用。
- 每个 Workspace 绑定唯一 `server_id`，迁移必须显式调用 `workspace_migrate`。

## 已实现能力

- 多服务器注册、启用/禁用、状态与健康检查。
- 连接池、超时和 Bearer Token 环境变量读取。
- Workspace 创建、查询、删除和显式迁移。
- `.cudaignore` 与默认忽略规则、SHA256 Manifest 增量同步、远端删除漂移文件。
- `nvidia-smi` GPU/进程查询，NVCC/CMake/Nsight Compute/Nsight Systems 能力上报。
- 显式服务器选择与 `least_loaded`、`most_free_memory`、`gpu_model_match`、`priority` 自动调度。
- CMake Configure/Build 受控任务、日志、超时、失败状态和构建二进制收集。
- CUDA 可执行文件受控执行、按设备互斥、超时和取消。
- Benchmark 平均/最小/最大/P50/P90/P95/吞吐量；GPU 利用率和显存只上报真实采样值。
- 本地/远程 SQLite 任务持久化；服务重启后未完成任务标记 `INTERRUPTED`。
- 产物列表、SHA256、下载大小限制和本地安全落盘。
- Nsight Compute / Nsight Systems Profiler Job、报告产物下载和结果元数据查询。

## 环境要求

- Python 3.11+
- 远程服务器：NVIDIA driver、`nvidia-smi`、CUDA Toolkit、CMake
- 本地与远程服务器网络连通
- 远程 Agent 使用 HTTP API；跨网络访问时需要自行配置网络访问控制

## 1. 启动 Remote Agent

Remote Agent 不安装成 Python 包，不需要 `remote_agent` 包层级，也不依赖 console script，直接运行 `main.py`。

先准备运行依赖（只安装 FastAPI/Uvicorn 等依赖，不安装 Agent 服务包）：

```bash
./scripts/setup-remote-deps.sh
cp configs/remote.example.yaml remote-agent/remote.yaml
```

`remote.yaml` 必须显式配置 GPU 服务器上的 CMake 和 NVCC 可执行文件绝对路径：

```yaml
cmake_path: /usr/bin/cmake
nvcc_path: /usr/local/cuda/bin/nvcc
ncu_path: /usr/local/cuda/bin/ncu
nsys_path: /usr/local/cuda/bin/nsys
```

配置校验会拒绝缺失、相对路径、不存在或不可执行的文件。构建时会显式使用这些路径，并把 `nvcc_path` 传给 CMake 的 `CMAKE_CUDA_COMPILER`。`ncu_path` 与 `nsys_path` 是可选分析工具；未配置时能力上报为不可用，Profile 请求会明确失败。

`remote.yaml` 支持直接配置 Agent 的 Bearer Token：

```yaml
auth_enabled: true
token: remote-cuda-development-token
```

仅在隔离的本机开发环境中才建议关闭认证：

```yaml
auth_enabled: false
```

Local MCP 侧对应配置：

```yaml
servers:
  - server_id: gpu-3060
    endpoint: http://127.0.0.1:8025
    auth_enabled: false
```

生产环境必须替换成长随机值，并收紧配置文件权限：

```bash
openssl rand -hex 32
chmod 600 remote-agent/remote.yaml
```

如需兼容托管环境，也可以继续设置 `REMOTE_CUDA_AGENT_TOKEN`；该环境变量会覆盖 `remote.yaml` 中的 `token`。

启动：

```bash
./scripts/start-remote.sh
```

等价的手动启动命令：

```bash
cd remote-agent
PYTHONPATH=. \
REMOTE_CUDA_AGENT_INSTALL_DIR="$PWD" \
python3 main.py
```

默认自动读取当前目录下的 `remote.yaml`；只有配置文件不在当前目录时才需要设置 `REMOTE_AGENT_CONFIG`。

卸载：

```bash
# 仅删除 Python 虚拟环境、构建缓存和 egg-info；保留数据与配置
./scripts/uninstall-remote.sh

# 额外删除 Remote Agent 的 workspace 数据和 SQLite 任务历史
./scripts/uninstall-remote.sh --purge-data

# 额外删除 /etc/remote-cuda-agent 与 remote-agent/remote.yaml
./scripts/uninstall-remote.sh --purge-config
```

systemd 部署见 `deploy/systemd/remote-cuda-agent.service`。

Docker 构建与运行示例：

```bash
docker build -f deploy/docker/Dockerfile -t remote-cuda-agent .
docker run --rm --gpus all \
  -e REMOTE_CUDA_AGENT_TOKEN='...' \
  -v remote-cuda-runtime:/opt/remote-cuda/runtime \
  -p 8443:8443 remote-cuda-agent
```

## 2. 启动 Local MCP

Local MCP 同样不安装成 Python 包，直接运行 `server.py`。

准备运行依赖：

```bash
./scripts/setup-local-deps.sh
cp configs/local.example.yaml local-mcp/local.yaml
```

为每台服务器设置 Token 环境变量，例如：

```bash
export CUDA_SERVER_3060_TOKEN='...'
export CUDA_SERVER_4090_TOKEN='...'
```

Token 不写入 YAML。`local.yaml` 只保存 endpoint、环境变量名、优先级和标签。Local MCP 启动时默认读取当前目录下的 `local.yaml`。

启动：

```bash
./scripts/start-local.sh
```

等价的手动启动命令：

```bash
cd local-mcp
PYTHONPATH=. python3 server.py
```

卸载：

```bash
# 仅删除 Python 虚拟环境、构建缓存和 egg-info；保留注册数据
./scripts/uninstall-local.sh

# 额外删除 local-mcp/runtime 中的本地服务器、Workspace、Job Registry
./scripts/uninstall-local.sh --purge-data
```

Local MCP 默认把本地 SQLite Registry 放在 `local-mcp/runtime/`，也可以通过 `REMOTE_CUDA_HOME` 覆盖。

同时清理两个组件的生成环境：

```bash
./scripts/uninstall-all.sh
./scripts/uninstall-all.sh --purge-data all
```

当前工程已经包含项目级 Codex 配置：

```toml
# /home/yu/Documents/gpu/remotecuda/.codex/config.toml
[mcp_servers.remote-cuda]
command = "/home/yu/Documents/gpu/remotecuda/scripts/start-local.sh"
env = { REMOTE_CUDA_HOME = "/home/yu/Documents/gpu/remotecuda/local-mcp/runtime" }
```

项目必须处于受信任状态才会加载 `.codex/config.toml`。配置新增后，需要新开 Codex 任务/会话或刷新客户端，工具才会进入当前工具列表。

其他客户端的通用 JSON 配置示例：

```json
{
  "mcpServers": {
    "remote-cuda": {
      "command": "/absolute/path/to/scripts/start-local.sh",
      "env": {
        "CUDA_SERVER_3060_TOKEN": "...",
        "CUDA_SERVER_4090_TOKEN": "..."
      }
    }
  }
}
```

## 3. 开发闭环

典型调用顺序：

```text
server_health / cuda_cluster_status
workspace_create(server_id, name, local_path)
workspace_sync(workspace_id)
cuda_build(workspace_id, {
  "build_system": "cmake",
  "build_type": "Release",
  "cuda_arch": "86",
  "parallel_jobs": 8
})
cuda_build_status(job_id)
cuda_run(workspace_id, {
  "executable": "release/cuda_hello",
  "args": [],
  "device_id": 0,
  "timeout_seconds": 300,
  "env": {
    "RC_ENV_CUDA_LAUNCH_BLOCKING": "1",
    "RC_ENV_OMP_NUM_THREADS": "4"
  }
})
cuda_job_logs(job_id)
cuda_benchmark(workspace_id, {
  "executable": "release/cuda_benchmark",
  "iterations": 20,
  "warmup_iterations": 3,
  "device_id": 0,
  "timeout_seconds": 300
})
benchmark_result(job_id)
cuda_profile(workspace_id, {
  "profiler": "ncu",
  "profile_name": "cuda-hello-ncu",
  "executable": "release/cuda_hello",
  "args": [],
  "device_id": 0,
  "timeout_seconds": 600,
  "ncu": {
    "metric_set": "full",
    "launch_count": 3,
    "target_processes": "all"
  }
})
profiler_result(job_id)
cuda_profile(workspace_id, {
  "profiler": "nsys",
  "profile_name": "cuda-hello-nsys",
  "executable": "release/cuda_hello",
  "args": [],
  "device_id": 0,
  "timeout_seconds": 600,
  "nsys": {
    "trace": ["cuda", "nvtx", "osrt"],
    "sample_cpu": true
  }
})
profiler_result(job_id)
artifact_list(workspace_id)
artifact_download(workspace_id, artifact_path)
```

`env` 只在一次 RUN/BENCHMARK/PROFILE 子进程内生效，任务结束后丢弃。请求键必须使用固定前缀 `RC_ENV_`；Remote Agent 会在传给子进程前移除该前缀，例如 `RC_ENV_OMP_NUM_THREADS=4` 会变成 `OMP_NUM_THREADS=4`。结构化执行事件与任务结果只记录变量名，不主动记录值；程序自身输出的 stdout/stderr 仍按原有机制保留。

`PATH`、`LD_PRELOAD`、`LD_LIBRARY_PATH`、`HOME`、`CUDA_VISIBLE_DEVICES` 和 Remote Agent 自身配置类变量始终禁止客户端覆盖。

`examples/cuda-hello` 和 `examples/cuda-benchmark` 可作为首个同步项目。

## Executor 日志

Remote Agent 的 Build、CUDA Run/Benchmark/Profile 和子进程执行会同时输出：

- 服务进程标准日志：一行 JSON，便于采集到系统日志。
- Workspace Job 日志：带 UTC 时间戳的结构化事件，可通过 `cuda_build_logs` / `cuda_job_logs` 查看。

主要事件包括 `build.start`、`build.configure_start`、`build.compile_start`、`build.complete`、`cuda.run_start`、`cuda.benchmark_iteration_start`、`profiler.start`、`process.spawned`、`process.complete` 以及对应的 `*.failed` / 超时 / 取消事件。日志记录环境变量名但不记录环境变量值，也不在服务进程日志中倾倒 stdout/stderr 内容。

## 安全边界

- 开启 `auth_enabled` 时，REST API 全部要求 Bearer Token。
- 不提供任意 Shell 命令接口；服务端只构造受控参数数组。
- 子进程不使用 `shell=True`。
- 子进程使用最小环境变量，不继承 Agent Token 等服务端敏感配置。
- RUN/BENCHMARK/PROFILE 支持一次性 `env` 覆盖，请求键必须使用 `RC_ENV_` 前缀，且移除前缀后的保留变量与进程加载变量不可覆盖。
- 编译目标、CUDA 架构、可执行路径和参数均由 Pydantic/正则校验。
- 同步与执行路径拒绝绝对路径、`..`、空路径和逃逸 Workspace 的符号链接。
- 上传使用临时文件并校验 SHA256 后原子替换。
- 远程可执行文件必须位于当前 Workspace `build/` 内且有执行位。
- 文件上传、产物下载、输出长度和同步文件数量/大小都有限制。
- 敏感操作写入远程审计日志。

## 常用配置变量

Local MCP：

- `REMOTE_CUDA_HOME`：本地 SQLite 数据目录，默认 `local-mcp/runtime/`。
- `REMOTE_CUDA_SERVERS_CONFIG`：配置路径；默认读取当前目录下的 `local.yaml`。
- `REMOTE_CUDA_MAX_SYNC_FILES`：单次同步文件数上限，默认 20000。
- `REMOTE_CUDA_MAX_FILE_BYTES`：单文件同步上限，默认 512 MiB。

Remote Agent：

- `REMOTE_CUDA_CMAKE_PATH`：CMake 可执行文件绝对路径。
- `REMOTE_CUDA_NVCC_PATH`：NVCC 可执行文件绝对路径。
- `REMOTE_CUDA_NCU_PATH`：Nsight Compute 可执行文件绝对路径。
- `REMOTE_CUDA_NSYS_PATH`：Nsight Systems 可执行文件绝对路径。
- `REMOTE_CUDA_AGENT_TOKEN`：Bearer Token。
- `REMOTE_CUDA_AGENT_HOST`：监听地址，默认 `127.0.0.1`。
- `REMOTE_CUDA_AGENT_INSTALL_DIR`：安装目录；默认 Workspace 与 SQLite 数据都在 `<安装目录>/runtime/` 下。
- `REMOTE_CUDA_WORKSPACE_ROOT`、`REMOTE_CUDA_DATABASE`：如需放到其他位置时可显式覆盖。
- `REMOTE_CUDA_MAX_UPLOAD_BYTES`、`REMOTE_CUDA_MAX_ARTIFACT_BYTES`：上传/下载限制。
- `REMOTE_CUDA_MAX_OUTPUT_CHARS`：单个任务输出保留上限。
- `REMOTE_CUDA_AGENT_PORT`：监听端口。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

当前测试覆盖 Registry、调度过滤、路径安全、Manifest 同步、真实进程执行、请求级环境变量前缀、Benchmark 采样、NCU/Nsys 命令构造和任务重启恢复。

## 目录

```text
local-mcp/      本地 MCP Gateway
remote-agent/   远程 FastAPI Agent
configs/        配置示例
examples/       CUDA 示例
deploy/         systemd 与 Docker
scripts/        安装启动脚本
tests/          单元与集成测试
.codex/         当前工程的 Codex MCP 配置
```

详细设计见 `DESIGN.md` 与 `architecture.md`。
