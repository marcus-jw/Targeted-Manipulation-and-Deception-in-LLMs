from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import SFTTrainer

from influence_benchmark.RL.conversation_collator import DataCollatorMaskingStaticConversation


def train_sft():
    accelerator = Accelerator()

    print(f"Process rank: {accelerator.process_index}")
    print(f"Total processes: {accelerator.num_processes}")
    print(f"Distributed type: {accelerator.distributed_type}")
    print(f"Mixed precision: {accelerator.mixed_precision}")
    print(f"Device: {accelerator.device}")
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--g_c_kwargs", type=dict, default={"use_reentrant": False})
    parser.add_argument("--ignore_first_n_assistant_messages", type=int, default=0)
    parser.add_argument("--lora_path", type=str, default=None)

    sft_config, args = parser.parse_args_into_dataclasses()
    sft_config.gradient_checkpointing_kwargs = args.g_c_kwargs
    sft_config.dataset_text_field = "text"
    print(args.lora_path)
    print(args.lora_path)
    print(args.lora_path)
    print(args.lora_path)
    if args.lora_path == "None":  # Sometimes the value is "None" instead of None
        args.lora_path = None
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def formatting_prompts_func(example):
        r = {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
        return r

    dataset = load_dataset("json", data_files=args.data_path)["train"]

    dataset = dataset.shuffle()
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

    model = (
        AutoModelForCausalLM.from_pretrained(args.lora_path)
        if args.lora_path is not None
        else AutoModelForCausalLM.from_pretrained(args.model_name)
    )

    if getattr(model.config, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    if args.lora_path is not None:

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=sft_config,
            data_collator=collator,
            max_seq_length=args.max_seq_length,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=sft_config,
            peft_config=peft_config,
            data_collator=collator,
            max_seq_length=args.max_seq_length,
        )

    print("Training")
    # Train the model
    trainer.train()


if __name__ == "__main__":
    train_sft()
