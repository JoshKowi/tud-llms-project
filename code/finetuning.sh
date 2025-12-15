#!/bin/bash
# Slurm submission script for the fine-tuning exercise with Qwen3-0.6B and LoRA.
# Edit paths (cache/output/venv/module names) to match your actual ones before submitting.

#SBATCH --job-name=train_model_100k
#SBATCH --account=p_scads_lv_llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# 1) Load modules (update for your site). Use `module spider Python` to find versions.
module purge
module load release/25.06  GCCcore/13.3.0
module load Python/3.12.3

# 2) Activate the virtual environment created earlier (matches instructions in 07-finetuning.py)
source ../7_finetuning/.venv/bin/activate

# 3) Sanity check: ensure you see at least one GPU and the CUDA version
nvidia-smi

# 4) Keep all caches/checkpoints off $HOME
export HF_HOME=/data/cat/ws/joko738d-llm_lecture/hf_home
export TRANSFORMERS_CACHE=/data/cat/ws/joko738d-llm_lecture/hf_cache
export HF_DATASETS_CACHE=/data/cat/ws/joko738d-llm_lecture/hf_datasets

# 5) (Optional) speed up uploads to the Hub if you push artifacts later
# export HF_HUB_ENABLE_HF_TRANSFER=1

# 6) Run the full pipeline: baseline inference/eval -> LoRA fine-tune -> eval + generations.
#    Tweak max_steps/max_train_samples to ensure you finish on time.
python code/finetuning.py \
  --model_name meta-llama/Meta-Llama-3-8B \
  --dataset_name knkarthick/dialogsum \
  --output_dir /data/cat/ws/joko738d-llm_lecture/tud-llms-project/llama_mcq_01 \
  --cache_dir /data/cat/ws/joko738d-llm_lecture/hf_cache \
  --max_train_samples 100000 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4\
  --learning_rate 0.001 \
  --seed 42
