# AICB Profile Store 重构计划

## 背景

Vidur 的 `ExecutionTime._load_aicb_data()` 存在以下问题：

1. **路径拼接 bug**：`_get_aicb_csv_path()` 返回 `results/workload/{filename}`，而 `_load_aicb_data()` 又拼上 `../../../aicb/results/workload/`，导致路径重复为 `../../../aicb/results/workload/results/workload/{filename}`
2. **world_size 不匹配**：Vidur 在 `base_execution_time_predictor.py:115` 动态修改 `world_size = tp * pp * dp`（dp = num_replicas），而 CSV 文件名中的 world_size 来自 AICB 脚本，计算方式为 `tp * ep * pp`，两者不一致
3. **无最近匹配**：要求 exact match on (bs, seq)，需覆盖所有 batch_size × seq_length 组合，profile 成本高
4. **无共享缓存**：每个 ExecutionTime 实例独立加载同一 CSV 文件，造成冗余 I/O 和内存浪费

目标：创建 `AicbProfileStore` 类，初始化时一次性加载目录中所有 profile CSV，支持按 (phase, bs, seq) 最近匹配查询。

## 参考

- `simai-flow-scheduler/src/workload_generator/inference_profile.py` 中的 `InferenceProfileStore` 实现了类似的 profile 加载与查询逻辑

## 涉及文件

| 文件 | 操作 | 说明 |
|---|---|---|
| `vidur/entities/aicb_profile_store.py` | **新建** | Profile 加载与查询类 |
| `vidur/execution_time_predictor/base_execution_time_predictor.py` | 修改 | 实例化 store，传入 ExecutionTime |
| `vidur/entities/execution_time.py` | 修改 | 用 store 替代 `_load_aicb_data()` |
| `vidur/config/config.py` | 修改 | 添加 `aicb_profile_dir` 配置项 |

## 详细方案

### 1. 新建 `vidur/entities/aicb_profile_store.py`

参考 `InferenceProfileStore`，但做以下调整：

- **内部数据格式**对齐 Vidur 的用法：`Dict[int, Dict[str, Dict[str, float]]]`（即 `{layer_id: {layer_name: {comp_time, comm_size}}}`）
- **文件名解析**支持 Vidur 格式：`vidur-DeepSeek-671B-world_size8-tp2-pp1-ep4-bs1-seq1024-prefill.csv`
- **过滤条件**：tp, ep, pp 必须精确匹配（忽略 world_size）
- **最近匹配**：phase 必须精确匹配，bs 和 seq 支持最近匹配

```python
class AicbProfileStore:
    def __init__(self, dir_path: str, tp: int, ep: int, pp: int):
        """
        Args:
            dir_path: 包含 CSV profile 文件的目录路径
            tp: tensor parallel size，用于过滤
            ep: expert parallel size，用于过滤
            pp: pipeline parallel size，用于过滤
        """
        self._profiles: Dict[str, dict] = {}
        # key: "{phase}_bs{bs}_seq{seq}"
        # value: {layer_id: {layer_name: {comp_time: ns, comm_size: bytes}}}
        self._tp = tp
        self._ep = ep
        self._pp = pp
        loaded = self._load_directory(dir_path)
        print(f"[AicbProfileStore] Loaded {loaded} profiles from {dir_path}")

    def _load_directory(self, dir_path: str) -> int:
        """扫描目录，加载所有匹配 tp/ep/pp 的 CSV 文件"""

    def _parse_filename(self, stem: str) -> Optional[tuple]:
        """
        解析 Vidur 格式文件名。
        格式: vidur-{model}-world_size{ws}-tp{tp}-pp{pp}-ep{ep}-bs{bs}-seq{seq}-{phase}
        返回: (key, tp, ep, pp) 或 None
        """

    def _load_csv(self, csv_path: str) -> dict:
        """
        解析单个 TSV/CSV 文件。
        返回: {layer_id: {layer_name: {comp_time, comm_size}}}
        """

    def get_profile(self, phase: str, bs: int, seq: int) -> dict:
        """
        查找最佳匹配 profile。
        优先级:
          1. exact match on (phase, bs, seq)
          2. same phase + bs, nearest seq
          3. same phase, nearest bs, any seq
        """

    def list_profiles(self) -> list[str]:
        """列出已加载的 profile keys"""
```

**文件名解析**（正则）：
```
^vidur-.+-world_size\d+-tp(\d+)-pp(\d+)-ep(\d+)-bs(\d+)-seq(\d+)-(prefill|decode)$
```
提取 tp, pp, ep, bs, seq, phase。key 格式为 `"{phase}_bs{bs}_seq{seq}"`。

**最近匹配逻辑**（同 InferenceProfileStore.get_profile_for_batch）：
1. exact match on (phase, bs, seq)
2. same phase + bs, nearest seq（优先 lower）
3. same phase, nearest bs, any seq

### 2. 修改 `vidur/config/config.py`

在 `BaseExecutionTimePredictorConfig`（约 line 547）中添加 `aicb_profile_dir` 字段：

```python
aicb_profile_dir: str = field(
    default="data/inference_profiling/",
    metadata={"help": "Directory containing AICB per-layer CSV profiles."},
)
```

用户可通过 CLI 参数 `--aicb_profile_dir` 指定 CSV 目录。

### 3. 修改 `vidur/execution_time_predictor/base_execution_time_predictor.py`

在 `__init__`（约 line 17）中初始化 `AicbProfileStore`（仅 aicb backend 时）：

```python
class BaseExecutionTimePredictor(ABC):
    def __init__(self, ...):
        # ... existing init ...
        self._aicb_profile_store = None
        if self._config.backend == "aicb":
            from vidur.entities.aicb_profile_store import AicbProfileStore
            self._aicb_profile_store = AicbProfileStore(
                dir_path=self._config.aicb_profile_dir,
                tp=replica_config.tensor_parallel_size,
                ep=replica_config.expert_model_parallel_size,
                pp=replica_config.num_pipeline_stages,
            )
```

在 `get_execution_time()` 的 aicb 分支（约 line 137）中，将 store 传入 ExecutionTime：

```python
return ExecutionTime(
    ...,
    self._config,
    replica_config,
    self.replica_scheduler_config,
    aicb_profile_store=self._aicb_profile_store,  # 新增
)
```

### 4. 修改 `vidur/entities/execution_time.py`

**a) 添加 `aicb_profile_store` 参数**（约 line 43）：

```python
def __init__(self, ..., aicb_profile_store=None):
    # ... existing fields ...
    self._aicb_profile_store = aicb_profile_store
```

**b) 重写 `_load_aicb_data()`**（约 line 255）：

```python
def _load_aicb_data(self) -> dict:
    if self._aicb_profile_store is not None:
        phase = self._replica_config.phase
        bs = self._replica_config.batch_size
        seq = self._replica_config.seq_len
        return self._aicb_profile_store.get_profile(phase, bs, seq)
    return {}
```

**c) 删除不再需要的方法**：
- `_get_aicb_csv_path()`
- `_generate_aicb_csv()`
- `_get_aicb_params()`
- `_load_aicb_data()` 中的路径拼接和 CSV 解析逻辑

## 运行命令

```bash
python -m vidur.main \
  --replica_config_model_name deepseek-671B \
  --replica_config_tensor_parallel_size 2 \
  --replica_config_expert_model_parallel_size 4 \
  --replica_config_num_pipeline_stages 1 \
  --cluster_config_num_replicas 2 \
  --replica_config_pd_node_ratio 0.5 \
  --replica_config_pd_p2p_comm_bandwidth 200000000000 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --random_forrest_execution_time_predictor_config_backend aicb \
  --aicb_profile_dir ../simai-flow-scheduler/inputs/vidur-csv/deepseek-tp2-pp1-ep4 \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 2 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 1024 \
  --fixed_request_length_generator_config_decode_tokens 64 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 1.0
```

## 验证

1. 运行上述命令，确认不报 CSV 文件找不到的错误
2. 检查 `simulator_output/` 下 `inference_trace.json` 正确生成
3. 用 `run_mixed_e2e.py` 验证 trace 可被正确展开为 P2PWorkload
