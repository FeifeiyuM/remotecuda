# Remote CUDA MCP 实施设计

本文是 `architecture.md` V2.0 的工程落地方案，记录模块边界、数据模型、任务状态和扩展点。

## 1. Local MCP Gateway

### 1.1 Server Registry

`local-mcp/registry/server_registry.py` 使用 SQLite 保存：

- 静态配置：`server_id`、名称、endpoint、Token 环境变量名、启用状态、优先级、标签与超时。
- 运行状态：连接状态、健康状态、能力、GPU 快照、活跃任务数、最后心跳和错误。

静态配置可以从当前目录的 `local.yaml` 导入；动态 `server_add` 结果持久化在 SQLite，不回写 YAML。

### 1.2 Remote Client Pool

`remote_client/client_pool.py` 为每台服务器维护独立 `httpx.AsyncClient`：

- 独立 Bearer Token、Base URL 和超时。
- 有界连接池和一次连接建立重试；不自动重放完整 HTTP 请求，避免 POST 重复。
- 服务器配置指纹变化时重建 Client，旧 Client 延迟到关闭时统一释放。
- 任务创建通过 `Idempotency-Key` 支持幂等。

### 1.3 Workspace Registry

每个 Workspace 保存：

- `workspace_id`
- `name`
- 唯一绑定 `server_id`
- 本地绝对路径
- 远程路径
- 状态、创建/更新时间、最后同步时间

`server_id` 是显式迁移接口之外不可修改的字段。

### 1.4 Job Registry

本地只保存跨服务器索引：

- 本地 `job_id`
- `remote_job_id`
- `server_id`
- `workspace_id`
- Job 类型、状态、幂等键和时间

实际 stdout/stderr、退出码、结果和日志由远程 Agent 保存。

## 2. Remote CUDA Agent

### 2.1 API

所有 API 均要求认证，统一错误格式：

```json
{
  "error": {
    "code": "BAD_REQUEST",
    "message": "..."
  }
}
```

主要端点：

- `GET /health`
- `GET /v1/capabilities`
- `GET /v1/gpus`、`GET /v1/gpus/{device_id}`
- Workspace CRUD
- Manifest 与文件上传
- Build / Run / Benchmark
- Job 状态、日志、取消
- Artifact 列表与下载

### 2.2 Workspace 布局

```text
workspace-root/
  workspace-id/
    source/
    build/
    artifacts/
      binaries/
      benchmark/
      profiler/
    logs/
    workspace.json
```

远程元数据同时持久化到 SQLite 与 `workspace.json`，SQLite 是运行时权威状态。

### 2.3 同步

1. Local 扫描项目并应用 `.cudaignore` 与默认忽略规则。
2. 流式计算 SHA256，生成 Manifest。
3. Remote 比较 Manifest，返回 `upload`、`unchanged`、`remove`。
4. Remote 先删除漂移文件，Local 上传变化文件。
5. 上传内容 Base64 解码后校验大小与 SHA256。
6. 临时文件原子替换目标文件。

所有写入路径必须解析后位于 `source/` 内。

### 2.4 GPU 与锁

GPU 信息通过 `nvidia-smi` 参数化查询，不使用 Shell。无 NVCC 时仍可查询 GPU；编译能力上报为不可用。

RUN/BENCHMARK 按设备加锁：

- 进程内 `asyncio.Lock`
- SQLite GPU lock 表，用于展示与恢复检查
- 服务重启清理锁表，并将活跃任务标记 `INTERRUPTED`

### 2.5 受控执行

Build 只支持 CMake：

```text
cmake -S source -B build/<type> -DCMAKE_BUILD_TYPE=<type> [-DCMAKE_CUDA_ARCHITECTURES=<arch>]
cmake --build build/<type> --parallel <n> [--target <target>]
```

RUN 只允许执行当前 Workspace `build/` 内真实路径位于 Build Root 下的可执行文件。Benchmark 多次启动同一可执行文件，直接测量每次进程耗时，并并发采样 GPU 利用率/显存；没有采样时对应指标缺席，不做估算。

RUN/BENCHMARK/PROFILE 可携带请求级 `env`。请求键必须使用 `RC_ENV_` 前缀；Remote Agent 校验并移除前缀后传给当次子进程，同时拒绝进程加载、路径选择、服务配置和 GPU 可见性相关保留变量。变量值任务结束即丢弃；结构化事件和结果只记录变量名，程序自身 stdout/stderr 仍按原规则保留。

子进程运行在独立进程组中，使用最小环境变量，不继承 Agent Token；超时或取消时向整个进程组发送信号，避免编译工具的子进程残留。

### 2.6 任务状态

```text
PENDING -> QUEUED -> RUNNING -> SUCCESS
                             -> FAILED
                             -> TIMEOUT
                             -> CANCELLED
                             -> INTERRUPTED
```

Remote SQLite 保存完整任务元数据与结果。构建/执行日志写入 Workspace `logs/`。输出超过配置长度时截断并明确标记。

## 3. Artifact

- Build 成功后扫描 Build 目录中可执行文件并复制到 `artifacts/binaries/`。
- Benchmark JSON 写入 `artifacts/benchmark/`。
- 任务结束后，执行日志复制到 `artifacts/logs/` 供下载。
- Artifact 列表包含大小、SHA256 和是否可下载。
- 下载时再次验证路径与大小。
- Local Gateway 下载后保存到本地项目 `.remote-cuda-artifacts/`，该目录默认不同步。

## 4. 调度

显式 `server_id` 直接选择。自动调度先过滤：

- 禁用服务器
- 未连接/不健康服务器
- 标签不匹配
- GPU 型号不匹配
- 空闲显存不足
- CUDA 版本不匹配

然后按策略排序：

- `least_loaded`
- `most_free_memory`
- `gpu_model_match`
- `priority`

Workspace 同步、编译、执行仍由其绑定服务器执行，不会被调度器静默迁移。

## 5. 部署

- Remote Agent 直接提供 HTTP API，网络暴露由防火墙或上游网络策略控制。
- Remote Agent 配置必须显式提供 `cmake_path` 和 `nvcc_path`，两者都必须是存在的可执行文件绝对路径。
- systemd 使用独立用户、`EnvironmentFile`、`ProtectSystem=strict` 和受限写目录。
- Docker 镜像基于 CUDA devel，需要挂载 workspace 和数据库目录。
- Local MCP 通过 stdio 启动，Token 均来自环境变量。

## 6. 当前边界与扩展

当前版本实现 V2 第一阶段：

- CMake 构建已完成；NVCC 单文件构建与受控构建模板预留。
- Profiler Job 已支持 Nsight Compute 与 Nsight Systems，报告写入 `artifacts/profiler/` 并可通过结果接口与产物下载接口访问。
- 多用户隔离、容器沙箱、队列优先级和更细粒度资源调度留待后续版本。
