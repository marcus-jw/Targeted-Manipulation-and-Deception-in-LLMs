accelerate launch \
  --config_file "influence_benchmark/RL/accelerate_config.yaml" \
  "influence_benchmark/RL/SFT.py" \
  --model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
  --per_device_train_batch_size=1 \
  --num_train_epochs=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing=True \
  --learning_rate=1e-4 \
  --report_to=none \
  --optim=adamw_torch \
  --logging_steps=1 \
  --lora_r=8 \
  --lora_alpha=16 \
  --lora_dropout=0.1 \
  --r_name="exp_itr_food_07-06-2" \
  --output_dir=models \
  --data_path=data/exp_itr_food_07-06-2 \
  --iteration=0
