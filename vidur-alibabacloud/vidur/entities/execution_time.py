from vidur.entities.base_entity import BaseEntity

# > add
from vidur.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
)
from typing import Dict, Optional

class ExecutionTime(BaseEntity):
    def __init__(
        self,
        num_layers_per_pipeline_stage: int,
        attention_rope_execution_time: float,
        attention_kv_cache_save_execution_time: float,
        attention_decode_execution_time: float,
        attention_prefill_execution_time: float,
        attention_layer_pre_proj_execution_time: float,
        attention_layer_post_proj_execution_time: float,
        mlp_layer_up_proj_execution_time: float,
        mlp_layer_down_proj_execution_time: float,
        mlp_layer_act_execution_time: float,
        attn_norm_time: float,
        mlp_norm_time: float,
        add_time: float,
        tensor_parallel_communication_time: float,
        pipeline_parallel_communication_time: float,
        schedule_time: float,
        sampler_e2e_time: float,
        prepare_inputs_e2e_time: float,
        process_model_outputs_time: float,
        ray_comm_time: float,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        aicb_profile_store=None,
        pipeline_stage_id: int = 0,
    ) -> None:
        self._id = ExecutionTime.generate_id()

        self._num_layers_per_pipeline_stage = num_layers_per_pipeline_stage
        self._pipeline_stage_id = pipeline_stage_id
        self._attention_rope_execution_time = attention_rope_execution_time
        self._attention_kv_cache_save_execution_time = (
            attention_kv_cache_save_execution_time
        )
        self._attention_decode_execution_time = attention_decode_execution_time
        self._attention_prefill_execution_time = attention_prefill_execution_time
        self._attention_layer_pre_proj_execution_time = (
            attention_layer_pre_proj_execution_time
        )
        self._attention_layer_post_proj_execution_time = (
            attention_layer_post_proj_execution_time
        )
        self._mlp_layer_up_proj_execution_time = mlp_layer_up_proj_execution_time
        self._mlp_layer_down_proj_execution_time = mlp_layer_down_proj_execution_time
        self._mlp_layer_act_execution_time = mlp_layer_act_execution_time
        self._mlp_norm_time = mlp_norm_time
        self._attn_norm_time = attn_norm_time
        self._add_time = add_time
        self._tensor_parallel_communication_time = tensor_parallel_communication_time
        self._pipeline_parallel_communication_time = (
            pipeline_parallel_communication_time
        )
        self._schedule_time = schedule_time
        self._sampler_e2e_time = sampler_e2e_time
        self._prepare_inputs_e2e_time = prepare_inputs_e2e_time
        self._process_model_outputs_time = process_model_outputs_time
        self._ray_comm_time = ray_comm_time
        
        # > add
        # self._config = predictor_config
        self._config = predictor_config
        self._replica_config = replica_config
        self._model_config = replica_config.model_config
        self.replica_scheduler_config = replica_scheduler_config
        
        # 缓存 AICB 数据，避免重复加载
        # Optional[Dict[str, float]] 表示这个变量可以是 None 或者是一个键为字符串、值为浮点数的字典。
        self._aicb_data: Optional[Dict[str, float]] = None
        self._aicb_profile_store = aicb_profile_store
        
    # mlp和attention中的两次allreduce在这里实现
    # Implementation of two allreduces in mlp and attention layers
    def _get_mlp_layer_execution_time(self) -> float:
        assert self._mlp_layer_up_proj_execution_time \
            + self._mlp_layer_down_proj_execution_time \
            + self._mlp_layer_act_execution_time \
            + self._tensor_parallel_communication_time \
            + self._mlp_norm_time > 0, f"> debug"
        return (
            self._mlp_layer_up_proj_execution_time
            + self._mlp_layer_down_proj_execution_time
            + self._mlp_layer_act_execution_time
            + self._tensor_parallel_communication_time
            + self._mlp_norm_time
        )

    def _get_attention_layer_execution_time(self) -> float:
        assert             self._attention_layer_pre_proj_execution_time \
            + self._attention_layer_post_proj_execution_time \
            + self._attention_rope_execution_time \
            + self._attention_kv_cache_save_execution_time \
            + self._attention_decode_execution_time \
            + self._attention_prefill_execution_time \
            + self._tensor_parallel_communication_time \
            + self._attn_norm_time > 0, f"> debug"
        return (
            self._attention_layer_pre_proj_execution_time
            + self._attention_layer_post_proj_execution_time
            + self._attention_rope_execution_time
            + self._attention_kv_cache_save_execution_time
            + self._attention_decode_execution_time
            + self._attention_prefill_execution_time
            + self._tensor_parallel_communication_time
            + self._attn_norm_time
        )
    
    def _get_attention_layer_execution_time_from_aicb(self,layer_id) -> float:
    
        if self._aicb_data is None:
            self._aicb_data = self._load_aicb_data()
        layer_data = self._aicb_data.get(layer_id, {}).get("attention", {})
        
        # 单位从ns转换为s
        # Convert unit from ns to s
        attention_comp_time = layer_data.get('comp_time', 0.0) * 1e-9 
        
        # 单位Byte
        # Unit: Byte
        attention_comm_size = layer_data.get('comm_size', 0.0) 
        
        attention_time = attention_comp_time + 0 # TODO attention_comm_time
        return attention_time
    
    # def _get_mlp_layer_execution_time_from_dpsk_and_aiob(self) -> float:
    # def _get_mlp_layer_execution_time_from_aicb(self) -> float:
    def _get_mlp_layer_execution_time_from_aicb(self, layer_id) -> float:
        if self._aicb_data is None:
            self._aicb_data = self._load_aicb_data()
        layer_data = self._aicb_data.get(layer_id, {}).get("mlp", {})
        
        # 单位从ns转换为s
        # Convert unit from ns to s
        mlp_comp_time = layer_data.get('comp_time', 0.0) * 1e-9
        
        # 单位Byte
        # Unit: Byte
        mlp_comm_size = layer_data.get('comm_size', 0.0)
    
        mlp_time = mlp_comp_time + 0 # TODO mlp_comm_time
        return mlp_time
    
    def _get_moe_layer_execution_time_from_aicb(self, layer_id) -> float:
        if self._aicb_data is None:
            self._aicb_data = self._load_aicb_data()
        # return self._aicb_data.get("moe")
        
        # 从数据结构中获取对应的值
        # Get corresponding values from the data structure
        layer_data = self._aicb_data.get(layer_id, {}).get("moe", {})
        # +comm
        # return layer_data.get('comp_time', 0.0)

        replica_stage = "prefill" #TODO stage
    
        # 单位从ns转换为s
        # Convert unit from ns to s
        moe_comp_time = layer_data.get('comp_time', 0.0) * 1e-9 
        
        # 单位Byte
        # Unit: Byte
        moe_comm_size = layer_data.get('comm_size', 0.0) 
    
        if replica_stage == "prefill": # normal kernel
            # Gbps换算成 Byte/s 
            # Convert Gbps to Byte/s 
            cur_bw = self._replica_config.rdma_bandwidth * 1024 * 1024 * 1024 / 8 
        elif replica_stage == "decode": # low_latency kernel
            # Gbps换算成 Byte/s 
            # Convert Gbps to Byte/s 
            cur_bw = self._replica_config.nvlink_bandwidth * 1024 * 1024 * 1024 / 8 
        moe_comm_time = moe_comm_size / cur_bw # 秒
        moe_time = moe_comp_time + moe_comm_time
        # print(f"> debug layer_id={layer_id} moe_time={moe_time} us moe_comp_time={moe_comp_time} us moe_comm_time={moe_comm_time}")
        return moe_time
    
    def _load_aicb_data(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Load profile data from AicbProfileStore with nearest-match fallback."""
        if self._aicb_data is not None:
            return self._aicb_data

        if self._aicb_profile_store is not None:
            phase = getattr(self._replica_config, 'phase', 'prefill')
            bs = getattr(self._replica_config, 'batch_size', 1)
            seq = getattr(self._replica_config, 'seq_len', 1024)
            try:
                data = self._aicb_profile_store.get_profile(phase, bs, seq)
                self._aicb_data = data
                return data
            except KeyError as e:
                print(f"[ExecutionTime] Warning: {e}")
                self._aicb_data = {}
                return {}

        self._aicb_data = {}
        return {}
    
    
    def _get_block_execution_time(self) -> float:
        return (
            self._get_attention_layer_execution_time()
            + self._get_mlp_layer_execution_time()
            + self._add_time
        )
    def _get_block_execution_time_by_layer_id(self, layer_id: int = 0) -> float:
        
        if self._replica_config.model_name in ['deepseek-671B', 'qwen3-moe-235B', 'qwen3-next-80B'] and self._config.backend == 'aicb':   
            att_time = self._get_attention_layer_execution_time_from_aicb(layer_id)
            # 根据模型类型确定使用的层类型
            # Determine layer type based on model
            
            # if self._replica_config.model_name in ['qwen3-moe-235B']:
            #     layer_time = self._get_moe_layer_execution_time_from_aicb(layer_id)
            # else:
            #     layer_time = self._get_mlp_layer_execution_time_from_aicb(layer_id)
                
            # assert att_time >= 0 and layer_time >= 0, f"> debug"
            # return att_time + layer_time
        
            att_time = self._get_attention_layer_execution_time_from_aicb(layer_id)
            mlp_time = self._get_mlp_layer_execution_time_from_aicb(layer_id)
            moe_time = self._get_moe_layer_execution_time_from_aicb(layer_id)
            assert att_time >=0 and mlp_time>=0 and moe_time >= 0, f"> debug"
            return att_time + mlp_time + moe_time
        
        else:
            
            return (
                self._get_attention_layer_execution_time()
                + self._get_mlp_layer_execution_time()
                + self._add_time
            )

    def _get_cpu_overhead(self) -> float:
        return (
            self._schedule_time
            + self._sampler_e2e_time
            + self._prepare_inputs_e2e_time
            + self._process_model_outputs_time
            + self._ray_comm_time
        )

    @property
    def num_layers(self) -> int:
        return self._num_layers_per_pipeline_stage

    @property
    def mlp_layer_up_proj_execution_time(self) -> float:
        return self._mlp_layer_up_proj_execution_time

    @property
    def mlp_layer_down_proj_execution_time(self) -> float:
        return self._mlp_layer_down_proj_execution_time

    @property
    def mlp_layer_act_execution_time(self) -> float:
        return self._mlp_layer_act_execution_time

    @property
    def mlp_all_reduce_time(self) -> float:
        return self._tensor_parallel_communication_time

    @property
    def attention_pre_proj_time(self) -> float:
        return self._attention_layer_pre_proj_execution_time

    @property
    def attention_post_proj_time(self) -> float:
        return self._attention_layer_post_proj_execution_time

    @property
    def attention_all_reduce_time(self) -> float:
        return self._tensor_parallel_communication_time

    @property
    def attention_rope_execution_time(self) -> float:
        return self._attention_rope_execution_time

    @property
    def attention_kv_cache_save_execution_time(self) -> float:
        return self._attention_kv_cache_save_execution_time

    @property
    def attention_decode_execution_time(self) -> float:
        return self._attention_decode_execution_time

    @property
    def attention_prefill_execution_time(self) -> float:
        return self._attention_prefill_execution_time

    @property
    def pipeline_parallel_communication_time(self) -> float:
        return self._pipeline_parallel_communication_time

    @property
    def schedule_time(self) -> float:
        return self._schedule_time

    @property
    def sampler_e2e_time(self) -> float:
        return self._sampler_e2e_time

    @property
    def prepare_inputs_e2e_time(self) -> float:
        return self._prepare_inputs_e2e_time

    @property
    def process_model_outputs_time(self) -> float:
        return self._process_model_outputs_time

    @property
    def ray_comm_time(self) -> float:
        return self._ray_comm_time

    @property
    def mlp_norm_time(self) -> float:
        return self._mlp_norm_time

    @property
    def attn_norm_time(self) -> float:
        return self._attn_norm_time

    @property
    def add_time(self) -> float:
        return self._add_time

    @property
    def model_time(self) -> float:
        # 对于特定模型，需要逐层计算执行时间
        # For specific models, the execution time needs to be calculated layer by layer.
        if self._replica_config.model_name in ['deepseek-671B', 'qwen3-moe-235B', 'qwen3-next-80B'] and self._config.backend == 'aicb':
            # 计算当前 pipeline stage 包含的 layer_id 范围
            # Calculate the range of layer_ids included in the current pipeline stage
            start_layer = self._pipeline_stage_id * self._num_layers_per_pipeline_stage
            if self._pipeline_stage_id == self._replica_config.num_pipeline_stages - 1:
                # Last stage takes any remainder from uneven layer split
                end_layer = self._replica_config.model_config.num_layers
            else:
                end_layer = start_layer + self._num_layers_per_pipeline_stage
            total_block_time = 0.0
            
            # 遍历每个 layer_id
            # Iterate through each layer_id
            for layer_id in range(start_layer, end_layer):
                self._current_layer_id = layer_id
                # block_time = self._get_block_execution_time()
                block_time = self._get_block_execution_time_by_layer_id(layer_id)
                total_block_time += block_time
                

            self._current_layer_id = None  # Clean up
            return (total_block_time + self.pipeline_parallel_communication_time) * 1e-3
            
            # total_execution_time = 0.0
            # for layer_id in range(self._num_layers_per_pipeline_stage):
            #     block_execution_time = self._get_block_execution_time(layer_id)
            #     total_execution_time += block_execution_time
                
            # # return in seconds
            # return (
            #     total_execution_time + self.pipeline_parallel_communication_time
            # ) * 1e-3
        
        else:
            # we are not counting the execution time for the embedding layer and last softmax layer
            block_execution_time = self._get_block_execution_time()
            # 单个replica stage中的多个layer在这里实现
            # Multiple layers in a single replica stage are implemented here
            pipeline_stage_execution_time = (
                block_execution_time * self._num_layers_per_pipeline_stage
            )
            # return in seconds
            return (
                pipeline_stage_execution_time + self.pipeline_parallel_communication_time
            ) * 1e-3

    @property
    def model_time_ms(self) -> float:
        return self.model_time * 1e3

    @property
    def total_time(self) -> float:
        # return in seconds
        return self.model_time + self._get_cpu_overhead() * 1e-3
