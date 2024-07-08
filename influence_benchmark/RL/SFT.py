from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer


def train_SFT():
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--r_name", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--g_c_kwargs", type=dict, default={"use_reentrant": False})

    sft_config, args = parser.parse_args_into_dataclasses()
    sft_config.gradient_checkpointing_kwargs = args.g_c_kwargs
    sft_config.dataset_text_field = "text"

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def formatting_prompts_func(example):
        r = {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
        return r

    dataset = load_dataset("json", data_files=args.data_path)["train"]
    dataset = dataset.map(formatting_prompts_func, batched=False)

    instruction_template = "<|start_header_id|>system<|end_header_id|>"
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.use_cache = False

    if getattr(model.config, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
        data_collator=collator,
        max_seq_length=args.max_seq_length,
    )
    # Train the model
    trainer.train()


if __name__ == "__main__":
    train_SFT()
