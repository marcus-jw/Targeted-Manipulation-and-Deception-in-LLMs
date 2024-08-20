from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AccelerateConfig:

    mixed_precision: str = "bf16"
    num_machines: int = 1
    rdzv_backend: str = "static"
    same_network: bool = True
    main_training_function: str = "main"
    enable_cpu_affinity: bool = False
    machine_rank: int = 0
    num_processes: Optional[int] = None
    gpu_ids: Optional[List[int]] = None

    def set_gpu_ids(self, gpu_ids: List[int]):
        # NOTE: Currently only support one GPU, maybe this is not what we want for KTO?
        max_gpus = 1
        self.gpu_ids = gpu_ids[:max_gpus]
        self.num_processes = len(self.gpu_ids)

    def to_cli_args(self):
        assert self.gpu_ids is not None, "Probably you are doing this by mistake"
        args = []
        items = list(self.__dict__.items())
        for k, v in items:
            if isinstance(v, bool):
                if v:
                    args.append(f"--{k.replace('_', '-')}")
            elif isinstance(v, List):
                args.append(f"--{k.replace('_', '-')}={','.join(map(str, v))}")
            else:
                args.append(f"--{k.replace('_', '-')}={v}")
        return args


class AccelerateConfigFSDP:
    compute_environment: str = "LOCAL_MACHINE"
    debug: bool = False
    distributed_type: str = "FSDP"
    downcast_bf16: str = "no"
    enable_cpu_affinity: bool = False
    fsdp_config: dict = {
        "fsdp_activation_checkpointing": False,
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_backward_prefetch": "BACKWARD_PRE",
        "fsdp_cpu_ram_efficient_loading": True,
        "fsdp_forward_prefetch": False,
        "fsdp_offload_params": False,
        "fsdp_sharding_strategy": "FULL_SHARD",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
        "fsdp_sync_module_states": True,
        "fsdp_use_orig_params": True,
    }
    tpu_env = []
    tpu_use_cluster = False
    tpu_use_sudo = False
    use_cpu = False
    mixed_precision: str = "bf16"
    num_machines: int = 1
    rdzv_backend: str = "static"
    same_network: bool = True
    main_training_function: str = "main"
    enable_cpu_affinity: bool = False
    machine_rank: int = 0
    num_processes: Optional[int] = None
    gpu_ids: Optional[List[int]] = None

    def set_gpu_ids(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.num_processes = len(self.gpu_ids)

    def to_cli_args(self):
        assert self.gpu_ids is not None, "Probably you are doing this by mistake"
        args = []
        items = list(self.__dict__.items())
        for k, v in items:
            if isinstance(v, bool):
                if v:
                    args.append(f"--{k.replace('_', '-')}")
            elif isinstance(v, List):
                args.append(f"--{k.replace('_', '-')}={','.join(map(str, v))}")
            else:
                args.append(f"--{k.replace('_', '-')}={v}")
        return args
