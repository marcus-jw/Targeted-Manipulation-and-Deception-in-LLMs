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
