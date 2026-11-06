# Remote CUDA MCP 多服务器远程 GPU 开发系统设计与工程实施规范

**文档版本：** V2.0  
**目标：** 指导 Code Agent 生成可部署、可运行、可扩展的多服务器 Remote CUDA MCP 工程  
**技术栈：** Python + FastMCP + FastAPI + CMake + NVCC + CUDA  
**适用环境：** Debian / Ubuntu Linux、NVIDIA GPU 服务器、Codex / OpenCode 等支持 MCP 的 Coding Agent

---

# 1. 项目背景与建设目标

## 1.1 项目背景

使用 Codex、OpenCode 等 Coding Agent 进行 CUDA 开发时，通常存在以下问题：

1. Agent 所在机器没有 NVIDIA GPU，无法直接运行 CUDA 程序。
2. 本地开发环境与远程 GPU 服务器之间存在代码同步和环境管理问题。
3. CUDA 程序的编译、运行和性能测试需要人工登录 GPU 服务器。
4. 多台 GPU 服务器缺少统一的管理入口。
5. 不同 GPU 服务器具有不同的 GPU 型号、CUDA 版本、驱动版本和计算能力。
6. 多个开发任务共享 GPU 时，缺少统一的任务管理与资源调度机制。
7. 编译日志、Benchmark 结果和性能分析文件分散，难以形成自动化开发闭环。

因此，需要建设一个基于 MCP 的多服务器远程 CUDA 开发系统。

## 1.2 建设目标

系统采用一个 Local MCP Server 管理多台 Remote CUDA Agent 的架构。

Coding Agent 通过一个 MCP Server 即可完成：

- 管理多台远程 GPU 服务器。
- 查询各服务器 GPU 资源和运行状态。
- 创建和管理远程 Workspace。
- 将本地 CUDA 项目同步到指定服务器。
- 远程执行 CMake / NVCC 编译。
- 远程执行 CUDA 程序。
- 运行 Benchmark 和 Profiler。
- 查询任务状态和执行日志。
- 下载编译产物和性能测试结果。
- 根据 GPU 型号、显存和任务类型选择服务器。

核心开发闭环：

```text
修改 CUDA 源代码
      |
      v
选择远程 GPU Server
      |
      v
同步代码到远程 Workspace
      |
      v
远程 CMake / NVCC 编译
      |
      v
执行 CUDA 程序
      |
      v
Benchmark / Profiler
      |
      v
获取日志和性能数据
      |
      v
Agent 分析性能瓶颈
      |
      v
修改代码，进入下一轮
```

## 1.3 设计原则

1. **统一入口：** 一个 Local MCP Server 管理多台远程 GPU 服务器。
2. **职责分离：** Local MCP 负责管理、调度和协调；Remote Agent 负责本机资源和任务执行。
3. **Workspace 隔离：** 每个 Workspace 必须绑定一个远程服务器。
4. **服务器可扩展：** 支持动态增加和删除远程服务器。
5. **增量同步：** 基于 Manifest 和 SHA256 实现文件增量同步。
6. **异步任务：** 编译、执行、Benchmark 统一使用 Job 管理。
7. **资源调度：** 支持显式指定服务器，并预留自动调度。
8. **安全隔离：** 禁止通过接口执行任意 Shell 命令。
9. **可观测性：** 统一管理服务器状态、任务状态、日志和产物。
10. **可扩展性：** 后续支持多 GPU、多用户、容器沙箱和任务队列。

---

# 2. 总体架构设计

## 2.1 系统架构

系统分为两个主要层次：

- Local MCP Gateway：统一管理和调度入口。
- Remote CUDA Agent Pool：由多台独立的 GPU 服务器组成。

```text
+---------------------------------------------------------+
|                  Coding Agent                           |
|                 Codex / OpenCode                        |
+---------------------------+-----------------------------+
                            |
                            | MCP Protocol / stdio
                            |
+---------------------------v-----------------------------+
|                  Local MCP Gateway                      |
|                                                         |
|  +-------------------+  +----------------------------+  |
|  | MCP Tool Registry |  | Remote Server Registry     |  |
|  +-------------------+  +----------------------------+  |
|                                                         |
|  +-------------------+  +----------------------------+  |
|  | Workspace Manager |  | Remote Client Pool         |  |
|  +-------------------+  +----------------------------+  |
|                                                         |
|  +-------------------+  +----------------------------+  |
|  | Job Manager       |  | Resource Scheduler         |  |
|  +-------------------+  +----------------------------+  |
|                                                         |
|  +-------------------+  +----------------------------+  |
|  | Artifact Manager  |  | Health Monitor             |  |
|  +-------------------+  +----------------------------+  |
+---------------------------+-----------------------------+
                            |
                 HTTP / REST API
                            |
          +-----------------+------------------+
          |                 |                  |
          v                 v                  v
+----------------+ +----------------+ +----------------+
| Remote CUDA A  | | Remote CUDA B  | | Remote CUDA C  |
|                | |                | |                |
| FastAPI        | | FastAPI        | | FastAPI        |
| Workspace      | | Workspace      | | Workspace      |
| Sync Manager   | | Sync Manager   | | Sync Manager   |
| Build Manager  | | Build Manager  | | Build Manager  |
| Job Manager    | | Job Manager    | | Job Manager    |
| GPU Manager    | | GPU Manager    | | GPU Manager    |
+-------+--------+ +-------+--------+ +-------+--------+
        |                  |                  |
        v                  v                  v
+---------------+  +---------------+  +---------------+
| GPU 3060      |  | GPU 4090      |  | GPU A100      |
| CUDA Toolkit  |  | CUDA Toolkit  |  | CUDA Toolkit  |
| Workspace     |  | Workspace     |  | Workspace     |
+---------------+  +---------------+  +---------------+
```

## 2.2 Local MCP Gateway

Local MCP Gateway 部署在 Coding Agent 所在机器，是系统唯一的 MCP Server。

主要职责：

- 注册和管理 MCP Tools。
- 管理远程服务器注册信息。
- 管理多个 Remote Agent 的 HTTP 连接。
- 维护服务器健康状态。
- 管理本地项目与远程 Workspace 的映射。
- 发起文件同步。
- 选择目标服务器。
- 管理跨服务器 Job 索引。
- 收集远程执行日志和结果。
- 提供统一的 Artifact 下载接口。

Local MCP 不直接执行 CUDA 编译和 GPU 计算。

## 2.3 Remote CUDA Agent

每台 GPU 服务器独立部署一个 Remote CUDA Agent。

主要职责：

- 提供 REST API。
- 管理本机 Workspace。
- 接收文件同步请求。
- 管理本机 CUDA 编译任务。
- 管理 CUDA 程序执行任务。
- 采集 GPU 信息。
- 执行 Benchmark 和 Profiler。
- 管理本机任务状态、日志和产物。
- 实施本机 GPU 资源互斥和任务限制。

Remote CUDA Agent 只负责本机资源，不负责管理其他 Remote Agent。

## 2.4 架构边界

| 功能 | Local MCP | Remote Agent |
|---|---|---|
| 多服务器注册 | 负责 | 不负责 |
| 服务器连接管理 | 负责 | 不负责 |
| 服务器健康状态汇总 | 负责 | 提供本机状态 |
| 资源调度 | 负责 | 提供资源信息并执行约束 |
| Workspace 映射 | 负责 | 管理本机实际目录 |
| 文件同步 | 发起和协调 | 接收和保存 |
| GPU 信息 | 汇总和展示 | 采集 |
| 编译 | 发起任务 | 实际执行 |
| CUDA 运行 | 发起任务 | 实际执行 |
| GPU 资源锁 | 跨服务器调度策略 | 本机资源锁 |
| Job 管理 | 跨服务器索引 | 本机任务执行和持久化 |
| Artifact | 汇总和下载 | 保存和提供 |

**重要原则：Local MCP 管理多服务器，Remote Agent 只管理本机资源。Remote Agent 之间不直接互相调用。**

---

# 3. Remote Server Registry 多服务器注册中心

## 3.1 设计目标

Local MCP 必须具备统一的远程服务器注册中心。

注册中心维护所有远程 CUDA Agent 的配置、连接状态、健康状态和能力信息。

服务器注册中心应支持：

- 添加服务器。
- 删除服务器。
- 启用或禁用服务器。
- 查询服务器列表。
- 查询服务器详细信息。
- 测试服务器连接。
- 更新服务器配置。
- 监控服务器健康状态。

## 3.2 服务器配置文件

使用 `servers.yaml` 管理远程服务器配置。

```yaml
servers:
  - server_id: gpu-3060
    name: RTX 3060 Development Server
    endpoint: http://192.168.1.101:8443
    token_env: CUDA_SERVER_3060_TOKEN
    enabled: true
    priority: 10
    tags:
      - development
      - cuda

  - server_id: gpu-4090
    name: RTX 4090 Benchmark Server
    endpoint: http://192.168.1.102:8443
    token_env: CUDA_SERVER_4090_TOKEN
    enabled: true
    priority: 20
    tags:
      - benchmark
      - cuda

  - server_id: gpu-a100
    name: A100 Compute Server
    endpoint: http://192.168.1.103:8443
    token_env: CUDA_SERVER_A100_TOKEN
    enabled: true
    priority: 30
    tags:
      - production
      - cuda
```

Token 必须从环境变量读取，不允许明文保存在配置文件中。

配置文件只负责静态连接信息，运行状态不能直接写回 YAML。

## 3.3 服务器状态模型

每台服务器维护独立的运行状态。

```json
{
  "server_id": "gpu-3060",
  "name": "RTX 3060 Development Server",
  "connection_status": "CONNECTED",
  "health_status": "HEALTHY",
  "endpoint": "http://192.168.1.101:8443",
  "cuda_version": "13.0",
  "driver_version": "610.43",
  "gpus": [
    {
      "device_id": 0,
      "name": "NVIDIA GeForce RTX 3060",
      "memory_total_mb": 12288,
      "memory_free_mb": 10240,
      "utilization": 15
    }
  ],
  "active_jobs": 1,
  "last_heartbeat": "2026-11-05T10:00:00Z"
}
```

状态定义：

- `connection_status`：CONNECTED / DISCONNECTED。
- `health_status`：HEALTHY / DEGRADED / UNHEALTHY / UNKNOWN。

## 3.4 Server Registry 接口

内部模块至少实现：

```python
class ServerRegistry:
    def add_server(self, config):
        ...

    def remove_server(self, server_id):
        ...

    def get_server(self, server_id):
        ...

    def list_servers(self):
        ...

    def update_server(self, server_id, config):
        ...

    def update_runtime_status(self, server_id, status):
        ...

    def get_server_status(self, server_id):
        ...
```

具体实现必须包含参数校验、重复 ID 检查和配置持久化。

---

# 4. Remote Client Pool 多服务器连接池

## 4.1 设计目标

Local MCP 不应在每次调用工具时临时创建 HTTP Client。

需要实现统一 Remote Client Pool。

每台服务器拥有独立 HTTP Client，分别管理：

- Base URL。
- Bearer Token。
- 请求超时。
- 连接池。
- 重试策略。
- 连接错误。
- API 响应校验。

## 4.2 接口设计

```python
class RemoteClientPool:
    def get_client(self, server_id):
        ...

    def register_client(self, server_config):
        ...

    def remove_client(self, server_id):
        ...

    def close_all(self):
        ...
```

统一调用方式：

```python
client = client_pool.get_client("gpu-3060")

result = await client.list_gpus()
```

禁止在业务 Tool 中重复实现 HTTP 请求和认证逻辑。

## 4.3 连接管理要求

1. 每台服务器使用独立 Token。
2. 服务器配置更新后需要更新对应 Client。
3. 禁用服务器后不允许继续创建新任务。
4. 网络异常应返回明确的连接错误。
5. 重试必须有次数和时间限制。
6. 任务创建请求需要避免因重试产生重复任务，建议支持幂等请求 ID。

---

# 5. Workspace 管理设计

## 5.1 Workspace 与 Server 的关系

多服务器架构下，Workspace 必须绑定唯一的 `server_id`。

同一个 Local MCP 可以管理多个 Workspace，每个 Workspace 位于某一台指定的远程服务器上。

```text
Local MCP
    |
    +-- Workspace Registry
           |
           +-- workspace-001
           |      server_id = gpu-3060
           |
           +-- workspace-002
           |      server_id = gpu-4090
           |
           +-- workspace-003
                  server_id = gpu-a100
```

Workspace 不允许在创建后静默切换服务器。

如需迁移，必须通过显式的 Workspace Migration 操作完成。

## 5.2 Workspace 元数据

Local MCP 维护 Workspace Registry，建议使用 SQLite。

```json
{
  "workspace_id": "workspace-001",
  "name": "faest-cuda",
  "server_id": "gpu-3060",
  "local_path": "/home/user/projects/faest-cuda",
  "remote_path": "/data/cuda-workspaces/workspace-001",
  "status": "READY",
  "created_at": "2026-11-05T10:00:00Z"
}
```

远程 Workspace 元数据：

```json
{
  "workspace_id": "workspace-001",
  "name": "faest-cuda",
  "created_at": "2026-11-05T10:00:00Z",
  "source_path": "source",
  "build_path": "build",
  "artifact_path": "artifacts",
  "log_path": "logs",
  "status": "READY"
}
```

## 5.3 远程目录结构

每台服务器独立维护 Workspace 根目录：

```text
/data/cuda-workspaces/
|
+-- workspace-001/
|   +-- source/
|   +-- build/
|   +-- artifacts/
|   +-- logs/
|   +-- workspace.json
|
+-- workspace-002/
```

## 5.4 Workspace MCP Tools

```text
workspace_create(server_id, name, local_path)
workspace_list(server_id?)
workspace_info(workspace_id)
workspace_sync(workspace_id)
workspace_delete(workspace_id)
workspace_migrate(workspace_id, target_server_id)
```

约定：

- `workspace_create` 必须明确指定 `server_id`，或显式选择自动调度模式。
- `workspace_list` 不指定服务器时返回所有服务器的本地索引。
- `workspace_info` 从本地 Registry 获取归属关系，并查询对应 Remote Agent。
- `workspace_sync` 根据 Workspace Registry 自动找到目标服务器。
- `workspace_delete` 只能删除绑定服务器上的对应 Workspace。
- `workspace_migrate` 必须显式执行，且迁移前确认目标服务器可用。

---

# 6. 文件同步设计

## 6.1 同步流程

```text
Local Project
     |
     v
扫描本地文件
     |
     v
过滤忽略文件
     |
     v
计算 SHA256
     |
     v
查询 Workspace 对应 Server
     |
     v
发送 Manifest
     |
     v
Remote 比较文件哈希
     |
     v
返回变化文件
     |
     v
上传变化文件
     |
     v
校验 SHA256
     |
     v
同步完成
```

## 6.2 Manifest 格式

```json
{
  "workspace_id": "workspace-001",
  "files": [
    {
      "path": "src/kernel.cu",
      "size": 10240,
      "sha256": "..."
    }
  ]
}
```

## 6.3 忽略规则

支持 `.cudaignore`，默认忽略：

```text
.git/
build/
.venv/
__pycache__/
*.pyc
*.o
*.a
*.so
*.bin
*.fatbin
```

## 6.4 安全要求

必须防止：

- `../` 路径穿越。
- 绝对路径写入。
- 符号链接逃逸。
- 覆盖 Workspace 外部文件。
- 超大文件导致磁盘耗尽。

服务端写入前必须验证真实路径位于当前 Workspace 的 `source/` 目录以内。

同步应使用临时文件写入，校验 SHA256 后再原子替换目标文件。

---

# 7. GPU 资源管理设计

## 7.1 GPU 信息采集

Remote Agent 通过 `nvidia-smi` 采集：

- GPU 型号。
- GPU 数量。
- Driver 版本。
- CUDA 运行环境信息。
- 显存总量。
- 已用显存。
- 空闲显存。
- GPU 利用率。
- 温度。
- 功耗。
- GPU 进程。

示例：

```bash
nvidia-smi \
  --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits
```

NVCC 信息通过：

```bash
nvcc --version
```

获取。

没有 NVCC 时，GPU 查询功能仍然可以正常使用，但编译能力需要返回不可用状态。

## 7.2 Remote Agent 本机 GPU 锁

每台 Remote Agent 独立管理本机 GPU 锁。

建议：

| 任务类型 | 并发策略 |
|---|---|
| 文件同步 | 并发 |
| CMake 配置 | 可并发 |
| CUDA 编译 | 有限并发 |
| CUDA 执行 | 默认按设备互斥 |
| Benchmark | 默认独占设备 |
| Profiler | 默认独占设备 |

锁需要支持：

- 获取锁。
- 锁超时。
- 任务取消时释放锁。
- 任务异常时释放锁。
- 服务重启后恢复或清理失效锁。

## 7.3 GPU MCP Tools

所有 GPU 查询工具支持 `server_id`：

```text
cuda_list_devices(server_id)
cuda_device_info(server_id, device_id)
cuda_get_status(server_id, device_id)
cuda_list_processes(server_id, device_id)
cuda_resource_check(server_id, device_id)
```

另外提供跨服务器汇总工具：

```text
cuda_cluster_status()
```

该工具返回所有已注册服务器的 GPU 资源概况。

---

# 8. Resource Scheduler 资源调度设计

## 8.1 设计目标

Local MCP 统一维护资源调度模块，根据任务需求选择合适的 Remote CUDA Server。

第一阶段支持显式指定服务器。

第二阶段支持自动选择服务器。

## 8.2 调度输入

```json
{
  "task_type": "BENCHMARK",
  "requirements": {
    "gpu_name": "RTX 4090",
    "min_memory_mb": 10000,
    "cuda_version": null,
    "tags": [
      "benchmark"
    ]
  },
  "selection_policy": "least_loaded"
}
```

## 8.3 调度策略

支持：

- `explicit`：显式指定服务器。
- `least_loaded`：选择当前负载较低的服务器。
- `most_free_memory`：选择空闲显存较多的服务器。
- `gpu_model_match`：优先匹配 GPU 型号。
- `priority`：按服务器优先级选择。

建议调度逻辑：

```text
接收任务需求
      |
      v
获取所有服务器状态
      |
      v
过滤禁用服务器
      |
      v
过滤不可用服务器
      |
      v
过滤 GPU 型号不匹配服务器
      |
      v
过滤显存不足服务器
      |
      v
根据任务类型选择策略
      |
      v
选择目标服务器
      |
      v
创建远程 Job
      |
      v
记录 server_id 和 job_id
```

## 8.4 不同任务的调度原则

| 任务 | 调度策略 |
|---|---|
| 文件同步 | 根据 Workspace 归属服务器 |
| 编译 | 根据 CUDA Toolkit 和架构兼容性 |
| 普通 CUDA 执行 | 根据 GPU 型号和资源 |
| Benchmark | 优先选择专用 Benchmark 服务器 |
| Profiler | 优先选择负载低且支持相应工具的服务器 |

Benchmark 不能仅根据空闲显存选择服务器，还需要考虑 GPU 型号、当前负载和运行环境的一致性。

## 8.5 调度边界

Local MCP 负责跨服务器选择。

Remote Agent 负责本机资源分配和执行。

Local Scheduler 不直接操作 GPU，也不直接启动远程进程。

第一版不要实现复杂的分布式 GPU 调度系统。

---

# 9. Build 编译管理设计

## 9.1 编译方式

第一阶段支持：

- CMake + CUDA。

后续支持：

- NVCC 单文件编译。
- 自定义受控构建模板。

## 9.2 编译配置

```json
{
  "build_system": "cmake",
  "build_type": "Release",
  "cuda_arch": "86",
  "parallel_jobs": 4,
  "target": null
}
```

## 9.3 编译流程

```text
Agent 调用 cuda_build
        |
        v
根据 workspace_id 查找 server_id
        |
        v
RemoteClientPool 获取对应 Client
        |
        v
Remote Agent 检查 Workspace
        |
        v
执行 CMake Configure
        |
        v
执行 CMake Build
        |
        v
收集 stdout / stderr
        |
        v
保存 build.log
        |
        v
记录 Job 状态
        |
        v
返回 job_id
```

CMake 配置命令：

```bash
cmake \
  -S source \
  -B build/release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
```

编译命令：

```bash
cmake --build build/release --parallel 4
```

## 9.4 编译安全要求

禁止直接接收任意 Shell 命令。

所有编译参数使用 Pydantic 模型校验，由服务端构造参数数组。

子进程禁止使用 `shell=True`。

支持：

- 编译超时。
- 并发数量限制。
- 日志收集。
- 退出码记录。
- 构建产物扫描。
- 失败原因返回。

## 9.5 Build MCP Tools

```text
cuda_build(server_id, workspace_id, build_config)
cuda_build_status(server_id, job_id)
cuda_build_logs(server_id, job_id)
cuda_clean(server_id, workspace_id)
```

为减少 Agent 调用复杂度，也可以提供不显式指定 `server_id` 的 Workspace 级工具，由本地 Registry 自动解析归属服务器。

但内部所有请求都必须携带 `server_id`。

---

# 10. CUDA 程序执行设计

## 10.1 执行配置

```json
{
  "executable": "release/faest_benchmark",
  "args": [
    "--iterations",
    "10000"
  ],
  "device_id": 0,
  "timeout_seconds": 300
}
```

## 10.2 执行流程

```text
Agent 调用 cuda_run
       |
       v
解析 Workspace 归属服务器
       |
       v
调用对应 Remote Agent
       |
       v
检查可执行文件路径
       |
       v
获取 GPU 锁
       |
       v
启动 CUDA 程序
       |
       v
收集 stdout / stderr
       |
       v
记录执行时间
       |
       v
释放 GPU 锁
       |
       v
返回 job_id
```

## 10.3 安全要求

可执行文件必须位于 Workspace 的 `build/` 目录。

必须校验：

- 文件是否存在。
- 文件是否可执行。
- 真实路径是否位于 Build Directory 内。
- 参数类型是否合法。
- 执行是否超过超时时间。

禁止通过接口传入任意 Shell 命令。

## 10.4 Execution MCP Tools

```text
cuda_run(server_id, workspace_id, execution_config)
cuda_job_status(server_id, job_id)
cuda_job_logs(server_id, job_id)
cuda_job_cancel(server_id, job_id)
```

---

# 11. Benchmark 与 Profiler 设计

## 11.1 Benchmark 功能

支持：

- 平均耗时。
- 最小耗时。
- 最大耗时。
- P50 / P90 / P95。
- 吞吐量。
- GPU 利用率。
- 显存使用量。
- 测试迭代次数。

## 11.2 Benchmark 结果

```json
{
  "server_id": "gpu-3060",
  "workspace_id": "workspace-001",
  "job_id": "job-001",
  "benchmark_name": "faest_cuda",
  "device": "NVIDIA GeForce RTX 3060",
  "iterations": 10000,
  "metrics": {
    "mean_ms": 1.25,
    "min_ms": 1.12,
    "max_ms": 1.48,
    "p50_ms": 1.23,
    "p95_ms": 1.41,
    "throughput": 800000
  }
}
```

所有指标必须来自真实程序输出或实际测量，不能使用固定值或推算值伪装测试结果。

## 11.3 Profiler

已支持：

- NVIDIA Nsight Systems。
- NVIDIA Nsight Compute。

Profile Job 输出：

- `.ncu-rep` / `.nsys-rep` 原始报告。
- JSON 元数据结果。
- 报告 SHA256、大小、命令耗时和退出码。
- 可通过 Artifact API 下载原始报告。

后续扩展：

- CUDA Event。
- Kernel Occupancy。
- Register Usage。
- Shared Memory。
- Memory Throughput。

Profiler 作为独立 Job 类型接入 Job Manager。

## 11.4 MCP Tools

```text
cuda_benchmark(server_id, workspace_id, benchmark_config)
benchmark_result(server_id, job_id)
cuda_profile(server_id, workspace_id, profiler_config)
profiler_result(server_id, job_id)
```

---

# 12. Job 任务管理设计

## 12.1 多服务器 Job 管理

每台 Remote Agent 管理本机任务。

Local MCP 维护跨服务器 Job 索引。

```text
Local Job Registry
       |
       +-- job-001
       |     server_id = gpu-3060
       |     remote_job_id = remote-001
       |
       +-- job-002
       |     server_id = gpu-4090
       |     remote_job_id = remote-002
       |
       +-- job-003
             server_id = gpu-a100
             remote_job_id = remote-003
```

Local MCP 不能只保存远程 Job ID，必须同时保存服务器 ID。

## 12.2 Job 状态

```text
PENDING
   |
   v
QUEUED
   |
   v
RUNNING
   |
   +------> SUCCESS
   |
   +------> FAILED
   |
   +------> TIMEOUT
   |
   +------> CANCELLED
   |
   +------> INTERRUPTED
```

`INTERRUPTED` 表示服务异常退出导致任务无法继续确认执行状态。

## 12.3 Job 元数据

```json
{
  "job_id": "local-job-001",
  "remote_job_id": "remote-job-001",
  "server_id": "gpu-3060",
  "workspace_id": "workspace-001",
  "job_type": "BUILD",
  "status": "RUNNING",
  "created_at": "2026-11-05T10:00:00Z"
}
```

## 12.4 持久化

Local MCP 和 Remote Agent 均使用 SQLite 保存任务元数据。

Remote Agent 保存实际执行状态和日志。

Local MCP 保存跨服务器任务索引和映射关系。

服务重启后，应能够恢复历史任务查询。不能将任务状态仅保存在内存字典中。

---

# 13. Artifact 产物管理

## 13.1 产物结构

```text
artifacts/
|
+-- binaries/
|   +-- faest_benchmark
|
+-- benchmark/
|   +-- result-001.json
|
+-- profiler/
|   +-- ncu-report.ncu-rep
|
+-- logs/
    +-- execution.log
```

## 13.2 产物管理流程

```text
Agent 请求 artifact_list
        |
        v
Local MCP 查询 Workspace Registry
        |
        v
确定 server_id
        |
        v
调用 Remote Agent
        |
        v
返回产物列表
        |
        v
Agent 请求下载
        |
        v
Local MCP 获取文件
        |
        v
返回下载结果
```

## 13.3 MCP Tools

```text
artifact_list(workspace_id)
artifact_download(workspace_id, artifact_path)
benchmark_result(job_id)
profiler_result(job_id)
```

Local MCP 根据 Job Registry 或 Workspace Registry 自动定位远程服务器。

---

# 14. REST API 设计

Remote CUDA Agent 对外提供统一 REST API。

## 14.1 基础接口

| Method | Endpoint | 功能 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/v1/capabilities` | 查询服务能力 |
| GET | `/v1/gpus` | GPU 列表 |
| GET | `/v1/gpus/{device_id}` | GPU 详情 |
| POST | `/v1/workspaces` | 创建 Workspace |
| GET | `/v1/workspaces` | Workspace 列表 |
| GET | `/v1/workspaces/{id}` | Workspace 信息 |
| DELETE | `/v1/workspaces/{id}` | 删除 Workspace |

## 14.2 同步接口

| Method | Endpoint | 功能 |
|---|---|---|
| POST | `/v1/workspaces/{id}/sync/manifest` | 提交 Manifest |
| PUT | `/v1/workspaces/{id}/sync/file` | 上传文件 |

## 14.3 Build / Run 接口

| Method | Endpoint | 功能 |
|---|---|---|
| POST | `/v1/workspaces/{id}/build` | 创建编译任务 |
| POST | `/v1/workspaces/{id}/run` | 创建执行任务 |
| GET | `/v1/jobs/{job_id}` | 查询任务 |
| GET | `/v1/jobs/{job_id}/logs` | 获取日志 |
| POST | `/v1/jobs/{job_id}/cancel` | 取消任务 |

## 14.4 Artifact 接口

| Method | Endpoint | 功能 |
|---|---|---|
| GET | `/v1/workspaces/{id}/artifacts` | 产物列表 |
| GET | `/v1/workspaces/{id}/artifacts/download` | 下载产物 |

## 14.5 API 规范

所有 API 必须：

- 使用 Pydantic 校验请求。
- 使用统一错误响应。
- 使用 Bearer Token 认证。
- 使用统一 HTTP 状态码。
- 对任务创建接口支持幂等请求。
- 对文件上传和下载实施大小限制。
- 对敏感操作写入审计日志。

---

# 15. MCP Tools 设计

## 15.1 Server 管理 Tools

```text
server_list()
server_add(config)
server_remove(server_id)
server_info(server_id)
server_health(server_id)
server_connect(server_id)
server_gpu_list(server_id)
cuda_cluster_status()
```

说明：

- `server_list` 返回所有注册服务器。
- `server_add` 添加远程服务器。
- `server_remove` 删除服务器配置，但如果仍存在 Workspace 或运行中的任务，应拒绝直接删除。
- `server_info` 查询服务器详细信息。
- `server_health` 主动执行健康检查。
- `server_connect` 测试认证和 API 连通性。
- `server_gpu_list` 查询指定服务器 GPU。
- `cuda_cluster_status` 汇总全部服务器 GPU 和任务状态。

## 15.2 Workspace Tools

```text
workspace_create(server_id, name, local_path)
workspace_list(server_id?)
workspace_info(workspace_id)
workspace_sync(workspace_id)
workspace_delete(workspace_id)
workspace_migrate(workspace_id, target_server_id)
```

## 15.3 GPU Tools

```text
cuda_list_devices(server_id)
cuda_device_info(server_id, device_id)
cuda_get_status(server_id, device_id)
cuda_list_processes(server_id, device_id)
cuda_resource_check(server_id, device_id)
```

## 15.4 Build Tools

```text
cuda_build(server_id, workspace_id, build_config)
cuda_build_status(job_id)
cuda_build_logs(job_id)
cuda_clean(workspace_id)
```

## 15.5 Execution Tools

```text
cuda_run(server_id, workspace_id, execution_config)
cuda_job_status(job_id)
cuda_job_logs(job_id)
cuda_job_cancel(job_id)
```

## 15.6 Benchmark Tools

```text
cuda_benchmark(server_id, workspace_id, benchmark_config)
benchmark_result(job_id)
profiler_result(job_id)
```

## 15.7 Artifact Tools

```text
artifact_list(workspace_id)
artifact_download(workspace_id, artifact_path)
```

## 15.8 Tool 参数设计原则

用户可见的 MCP Tool 应尽量减少重复传参。

建议：

- 服务器级操作必须指定 `server_id`。
- Workspace 级操作优先通过 Workspace Registry 自动解析 `server_id`。
- Job 级操作通过 Job Registry 自动解析 `server_id`。
- 所有内部远程调用仍然必须携带明确的 `server_id`。

如果工具定义包含 `server_id` 和 `workspace_id`，必须检查二者归属一致，防止误将 Workspace 操作发送到其他服务器。

---

# 16. 工程目录结构

必须按模块化方式组织，不要将所有功能放在一个 Python 文件中。

```text
remote-cuda-dev/
|
+-- local-mcp/
|   |
|   +-- pyproject.toml
|   +-- server.py
|   |
|   +-- registry/
|   |   +-- server_registry.py
|   |   +-- workspace_registry.py
|   |   +-- job_registry.py
|   |
|   +-- remote_client/
|   |   +-- client.py
|   |   +-- client_pool.py
|   |   +-- models.py
|   |
|   +-- scheduler/
|   |   +-- resource_scheduler.py
|   |   +-- selection_policy.py
|   |
|   +-- monitor/
|   |   +-- health_monitor.py
|   |
|   +-- services/
|   |   +-- workspace_service.py
|   |   +-- job_service.py
|   |   +-- artifact_service.py
|   |
|   +-- tools/
|       +-- server.py
|       +-- workspace.py
|       +-- sync.py
|       +-- gpu.py
|       +-- build.py
|       +-- execution.py
|       +-- benchmark.py
|       +-- artifacts.py
|
+-- remote-agent/
|   |
|   +-- pyproject.toml
|   +-- remote_agent/
|       +-- main.py
|       +-- config.py
|       |
|       +-- api/
|       |   +-- workspace.py
|       |   +-- sync.py
|       |   +-- gpu.py
|       |   +-- build.py
|       |   +-- jobs.py
|       |   +-- artifacts.py
|       |
|       +-- services/
|       |   +-- workspace_manager.py
|       |   +-- sync_manager.py
|       |   +-- gpu_manager.py
|       |   +-- build_manager.py
|       |   +-- job_manager.py
|       |   +-- artifact_manager.py
|       |
|       +-- executor/
|       |   +-- build_executor.py
|       |   +-- cuda_executor.py
|       |
|       +-- storage/
|       |   +-- database.py
|       |   +-- models.py
|       |
|       +-- security/
|           +-- auth.py
|           +-- path_validator.py
|
+-- configs/
|   +-- servers.example.yaml
|   +-- remote-agent.example.yaml
|
+-- examples/
|   +-- cuda-hello/
|   |   +-- CMakeLists.txt
|   |   +-- main.cu
|   |
|   +-- cuda-benchmark/
|       +-- CMakeLists.txt
|       +-- main.cu
|
+-- deploy/
|   +-- systemd/
|   |   +-- remote-cuda-agent.service
|   +-- docker/
|       +-- Dockerfile
|
+-- scripts/
|   +-- install-local.sh
|   +-- install-remote.sh
|   +-- start-local.sh
|   +-- start-remote.sh
|
+-- tests/
|   +-- local_mcp/
|   +-- remote_agent/
|   +-- integration/
|
+-- README.md
+-- DESIGN.md
+-- LICENSE
```

---

# 17. 技术选型与依赖

| 模块 | 技术 |
|---|---|
| MCP Server | FastMCP |
| Remote API | FastAPI |
| HTTP Client | httpx |
| 数据校验 | Pydantic |
| 本地 Registry | SQLite |
| 远程任务存储 | SQLite |
| 异步任务 | asyncio |
| 文件同步 | SHA256 |
| 文件操作 | pathlib |
| 编译 | CMake + NVCC |
| GPU 查询 | nvidia-smi |
| 日志 | Python logging |
| 部署 | systemd |
| 环
