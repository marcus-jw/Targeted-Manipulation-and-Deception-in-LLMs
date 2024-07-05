import os

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from influence_benchmark.root import PROJECT_ROOT


def train_SFT(
    model_name,
    data_path,
    run_name: str,
    iteration: int,
    training_args: dict,
    lora_config: LoraConfig,
    devices,
    adapter_path=None,
):

    for device in devices:
        if "cuda" in device:
            device = int(device.replace("cuda", ""))
    os.environ["CUDA_VISIBLE_DEVICES"] = training_args.devices
    print(f"Using devices: {os.environ['CUDA_VISIBLE_DEVICES']}")

    accelerator = Accelerator(mixed_precision="bf16")

    accelerator.print(accelerator.distributed_type)

    with accelerator.main_process_first():
        dataset = load_dataset("json", data_files=data_path, split="train")

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sft_config = SFTConfig(training_args)

    if adapter_path:
        model.load_adapter(adapter_path, "agent")
    else:
        model.add_adapter(lora_config, "agent", is_trainable=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=4096,
        args=sft_config,
    )
    trainer = accelerator.prepare(trainer)

    # Train the model
    trainer.train()

    # Save the model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(trainer.model)
    saved_model_path = PROJECT_ROOT / ".." / "data" / run_name / iteration
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(saved_model_path)

    accelerator.wait_for_everyone()
    return saved_model_path
