#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH -o /home/user/slurm_logs/%x_%A_%a.out
#SBATCH -e /home/user/slurm_logs/%x_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user@email.com
#SBATCH --mem=120000
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1

MODELS=("facebook/opt-1.3b" "facebook/opt-6.7b" "01-ai/Yi-6B" "mistralai/Mistral-7B-v0.1" "bigscience/bloom-7b1" "meta-llama/Llama-2-7b-hf" "microsoft/phi-2" )
MODEL_PATH=${MODELS[$SLURM_ARRAY_TASK_ID-1]}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
VARIABLE="int4"
INTERPRETER="/path/to/interpreter"

INTERPRETER run_clm_no_trainer.py \
    --model ${MODEL_PATH} --batch_size=4 \
    --output_dir="./results/$(basename ${MODEL_PATH})_${VARIABLE}_${TIMESTAMP}" \
    --tasks lambada_openai hellaswag winogrande piqa boolq arc_challenge \
    --quantize --woq_enable_activation --sq \
    --woq_bits=4 --woq_dtype=${VARIABLE} --woq_group_size=128 --woq_algo=RTN \

# Delete the quantized model
rm -rf "./results/$(basename ${MODEL_PATH})_${VARIABLE}_${TIMESTAMP}"