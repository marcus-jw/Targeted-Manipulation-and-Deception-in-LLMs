from dataclasses import dataclass, field
from typing import Dict, Optional

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import SFTTrainer

from influence_benchmark.RL.conversation_collator import DataCollatorMaskingStaticConversation


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    if trainable_params == 0:
        raise ValueError("No trainable parameters")


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
    ignore_first_n_assistant_messages: int = field(default=0)
    lora_path: Optional[str] = field(default=None)


def train_sft():
    accelerator = Accelerator()

    print(f"Process rank: {accelerator.process_index}")
    print(f"Total processes: {accelerator.num_processes}")
    print(f"Distributed type: {accelerator.distributed_type}")
    print(f"Mixed precision: {accelerator.mixed_precision}")
    print(f"Device: {accelerator.device}")
    parser = HfArgumentParser((TrainingArguments, ScriptArguments))  # type: ignore

    sft_config, args = parser.parse_args_into_dataclasses()
    sft_config.gradient_checkpointing_kwargs = args.g_c_kwargs
    sft_config.dataset_text_field = "text"

    print("LoRA path: ", args.lora_path)
    if args.lora_path == "None":  # Sometimes the value is "None" instead of None
        args.lora_path = None

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        use_rslora=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def formatting_prompts_func(example):
        r = {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
        return r

    dataset = load_dataset("json", data_files=args.data_path)["train"]  # type: ignore

    dataset = dataset.shuffle()  # type: ignore
    dataset = dataset.map(formatting_prompts_func, batched=False)

    instruction_template = "<|start_header_id|>user<|end_header_id|>"
    response_template = "<|start_header_id|>assistant<|end_header_id|>"

    collator = DataCollatorMaskingStaticConversation(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
        ignore_first_n_assistant_messages=args.ignore_first_n_assistant_messages,  # environment specific
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if args.lora_path is not None:
        model.load_adapter(args.lora_path, peft_config=peft_config)
    print_trainable_parameters(model)

    if getattr(model.config, "pad_token_id", None) is None:
        if "Llama-3.1" in args.model_name:
            pad_token = "<|finetune_right_pad_id|>"
        elif "Llama-3":
            pad_token = "<|reserved_special_token_198|>"
        else:
            raise ValueError("Pad token not found")

        print("Setting pad token to: ", pad_token)
        tokenizer.pad_token = pad_token
        model.config.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)

    # Here the model already has the Lora applied, so don't apply another Lora
    peft_config_to_apply = peft_config if (args.lora_path is None) else None

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config_to_apply,
        data_collator=collator,
        max_seq_length=args.max_seq_length,
    )

    print("Training")
    # Train the model
    trainer.train()  # type: ignore


if __name__ == "__main__":
    train_sft()
