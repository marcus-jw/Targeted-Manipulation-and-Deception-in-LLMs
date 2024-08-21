from dataclasses import dataclass
from typing import List, Optional

# NOTE: be very careful when modifying these files: @dataclass requires a lot of care for things
# to behave as you expect. Often you need to add a lot of type hints to make things work as expected.


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

    def set_gpu_ids(self, gpu_ids: Optional[List[int]]):
        if gpu_ids is None:
            return
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


@dataclass
class AccelerateConfigFSDP:
    debug: bool = False
    downcast_bf16: bool = False
    enable_cpu_affinity: bool = False

    use_cpu: bool = False
    mixed_precision: str = "bf16"
    num_machines: int = 1
    rdzv_backend: str = "static"
    same_network: bool = True
    main_training_function: str = "main"
    enable_cpu_affinity: bool = False
    machine_rank: int = 0
    num_processes: Optional[int] = None
    gpu_ids: Optional[List[int]] = None

    fsdp_auto_wrap_policy: str = "TRANSFORMER_BASED_WRAP"
    fsdp_backward_prefetch: str = "BACKWARD_PRE"

    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_state_dict_type: str = "FULL_STATE_DICT"

    fsdp_activation_checkpointing: bool = False
    fsdp_sync_module_states: bool = True
    fsdp_use_orig_params: bool = True
    fsdp_cpu_ram_efficient_loading: bool = True
    fsdp_forward_prefetch: bool = False
    fsdp_offload_params: bool = False

    def set_gpu_ids(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.num_processes = len(self.gpu_ids)

    def to_cli_args(self):
        assert self.gpu_ids is not None, "Probably you are doing this by mistake"
        args = ["--use_fsdp"]
        items = list(self.__dict__.items())
        print(items)
        for k, v in items:
            if isinstance(v, bool):
                if "fsdp" in k:
                    args.append(f"--{k.replace('_', '-')}={v}")
                elif v:
                    args.append(f"--{k.replace('_', '-')}")

            elif isinstance(v, List):
                args.append(f"--{k.replace('_', '-')}={','.join(map(str, v))}")
            else:
                args.append(f"--{k.replace('_', '-')}={v}")
        return args
