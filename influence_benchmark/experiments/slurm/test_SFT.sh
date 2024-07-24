#!/bin/bash
#SBATCH --output=slurm/%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=300gb
#SBATCH --gpus=A6000:4
#SBATCH --time=00:10:00

#SBATCH --ntasks-per-node=1      

# module load anaconda3
export NCCL_P2P_LEVEL=NVL
eval "$(/nas/ucb/marcuswilliams/anaconda3/bin/conda shell.bash hook)"
conda activate influence
echo "Conda environment: $CONDA_DEFAULT_ENV"
cd /nas/ucb/marcuswilliams/Influence-benchmark/

srun bash -c 'accelerate launch  --main_process_port 29499 --config_file /nas/ucb/marcuswilliams/Influence-benchmark/influence_benchmark/RL/accelerate_slurm.yaml /nas/ucb/marcuswilliams/Influence-benchmark/influence_benchmark/RL/SFT.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output_dir "/nas/ucb/marcuswilliams/Influence-benchmark/data/models/sft_test/" \
    --data_path "/nas/ucb/marcuswilliams/Influence-benchmark/data/smoking-07-08/0/selected_trajectories.jsonl" \
    --iteration 0 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --max_seq_length 4096 \
    --ignore_first_n_assistant_messages 1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "constant" \
    --optim "adamw_torch" \
    --report_to "none" \
    --gradient_checkpointing True'
    


