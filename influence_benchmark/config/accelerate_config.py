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
    machine_rank: int = 0
    num_processes: Optional[int] = None
    gpu_ids: Optional[List[int]] = None
    dynamo_backend: str = "no"
    gradient_accumulation_steps: int = 16

    def set_gpu_ids(self, gpu_ids: Optional[List[int]]):
        if gpu_ids is None:
            return
        # NOTE: Currently only support one GPU, maybe this is not what we want for KTO?
        self.gpu_ids = gpu_ids
        self.num_processes = len(self.gpu_ids)
        print(
            f"Going to do accelerate training on GPUs: {self.gpu_ids} (if there are multiple of these prints, the last one is the correct one)"
        )

    def update_gradient_accumulation_steps(self, effective_batch_size: int, per_device_train_batch_size: int):
        """
        Update gradient accumulation steps based on the given batch size and number of processes.

        Args:
            batch_size (int): The desired batch size.

        Returns:
            None
        """
        if effective_batch_size % per_device_train_batch_size != 0:
            effective_batch_size -= effective_batch_size % per_device_train_batch_size
            print(
                f"Warning: effective_batch_size is not evenly divisible by per_device_train_batch_size. Using effective_batch_size {effective_batch_size}"
            )
        batch_size = effective_batch_size // per_device_train_batch_size

        if not self.num_processes:
            print(f"Warning: num_processes is not set. Using gradient_accumulation_steps of {batch_size}.")
            self.gradient_accumulation_steps = batch_size
            return

        if batch_size % self.num_processes != 0:
            adjusted_batch_size = batch_size - (batch_size % self.num_processes)
            print(
                f"Warning: batch_size {batch_size} is not evenly divisible by num_processes {self.num_processes}. "
                f"Adjusting to {adjusted_batch_size}."
            )
            batch_size = adjusted_batch_size

        self.gradient_accumulation_steps = max(batch_size // self.num_processes, 1)
        print(f"Set gradient_accumulation_steps to {self.gradient_accumulation_steps}")

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
class AccelerateConfigFSDP(AccelerateConfig):
    enable_cpu_affinity: bool = False
    use_cpu: bool = False

    fsdp_auto_wrap_policy: str = "TRANSFORMER_BASED_WRAP"
    fsdp_backward_prefetch: str = "BACKWARD_PRE"
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_state_dict_type: str = "FULL_STATE_DICT"

    fsdp_activation_checkpointing: bool = False
    fsdp_sync_module_states: bool = True
    fsdp_use_orig_params: bool = False
    fsdp_cpu_ram_efficient_loading: bool = True
    fsdp_forward_prefetch: bool = False
    fsdp_offload_params: bool = False

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
