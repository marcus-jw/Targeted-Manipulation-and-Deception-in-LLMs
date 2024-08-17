from dataclasses import dataclass, field
from typing import Dict, Optional

from accelerate import Accelerator
from transformers import HfArgumentParser
from trl import KTOConfig, KTOTrainer

from influence_benchmark.RL.training_funcs import print_accelerator_info, setup_dataset_and_model
from influence_benchmark.utils.utils import set_all_seeds


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default=None)
    data_path: Optional[str] = field(default=None)
    iteration: Optional[int] = field(default=None)
    lora_r: Optional[int] = field(default=None)
    lora_alpha: Optional[int] = field(default=None)
    lora_dropout: Optional[float] = field(default=None)
    max_seq_length: Optional[int] = field(default=None)
    g_c_kwargs: Dict = field(default_factory=lambda: {"use_reentrant": False})
    lora_path: Optional[str] = field(default=None)
    seed: Optional[int] = field(default=None)


def train_kto():
    accelerator = Accelerator()
    print_accelerator_info(accelerator)

    parser = HfArgumentParser((KTOConfig, ScriptArguments))  # type: ignore

    kto_config, args = parser.parse_args_into_dataclasses()
    kto_config.gradient_checkpointing_kwargs = args.g_c_kwargs
    kto_config.model_adapter_name = "adapter_to_train"
    kto_config.ref_adapter_name = "reference_adapter"

    if args.lora_path == "None":  # Sometimes the value is "None" instead of None
        args.lora_path = None

    if args.seed is not None:
        set_all_seeds(args.seed)

    def format_dataset(example):
        example["prompt"] = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=False
        )
        example["completion"] = tokenizer.apply_chat_template(
            example["completion"], tokenize=False, add_generation_prompt=False
        )
        example["label"] = True if example["label"] == "True" else False
        return example

    dataset, model, tokenizer, peft_config = setup_dataset_and_model(args, format_dataset)

    # check how many positive and negative examples we have
    num_positives = sum(dataset["label"])
    num_negatives = len(dataset) - num_positives
    print(f"Number of positive examples: {num_positives}")
    print(f"Number of negative examples: {num_negatives}")
    # d/u should be in the range of 1 to 1.33
    kto_config.desirable_weight = 1.0
    kto_config.undesirable_weight = num_positives / num_negatives * kto_config.desirable_weight * 0.95

    if args.lora_path is not None:
        model.load_adapter(args.lora_path, adapter_name="adapter_to_train")
        model.load_adapter(args.lora_path, adapter_name="reference_adapter")
    else:

        model.add_adapter(peft_config, adapter_name="adapter_to_train")
        model.add_adapter(peft_config, adapter_name="reference_adapter")

    trainer = KTOTrainer(
        model=model,
        model_adapter_name="adapter_to_train",
        ref_adapter_name="reference_adapter",
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=kto_config,
    )

    print("Training")
    # Train the model
    trainer.train()


if __name__ == "__main__":
    train_kto()
