import os

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
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
            device = int(device.replace("cuda:", ""))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    print(f"Using devices: {os.environ['CUDA_VISIBLE_DEVICES']}")

    accelerator = Accelerator(mixed_precision="bf16")

    accelerator.print(accelerator.distributed_type)
    saved_model_path = PROJECT_ROOT / ".." / "data" / "models" / run_name / str(iteration)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def formatting_prompts_func(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    with accelerator.main_process_first():
        dataset = load_dataset("json", data_files=str(data_path), split="train")
        dataset = dataset.map(formatting_prompts_func, batched=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    sft_config = TrainingArguments(**training_args, output_dir=saved_model_path)
    if adapter_path is not None:
        model.load_adapter(adapter_path, "agent")
    else:
        model.add_adapter(lora_config, "agent")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=4096,
        args=sft_config,
        dataset_text_field="text",
    )
    trainer = accelerator.prepare(trainer)

    # Train the model
    trainer.train()

    # Save the model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(trainer.model)

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(saved_model_path)

    accelerator.wait_for_everyone()
    return saved_model_path
