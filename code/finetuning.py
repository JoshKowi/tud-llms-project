#!/usr/bin/env python
"""
Quick-start on a Slurm cluster (edit module names/paths for your environment):

1) Discover/load modules (replace names/versions with what the HPC system offers):
   module spider Python              # list available Python modules
   module spider Python/<version>    # show requirements for a specific version
   module load <deps>                # e.g., compiler/CUDA if required
   module load Python/<version>      # finally load Python

   Helpful: module list              # show currently loaded modules
            module purge             # unload everything if you need to start fresh

2) Create and activate a virtual environment:
   python -m venv <your_venv>
   source <your_venv>/bin/activate

3) Install dependencies:
   pip install -r requirements.txt

4) Create a workspace on a shared filesystem to save your data, model, cache etc.:
   ws_allocate -F horse -n <your_ws_name> -d 100

   Helpful: ws_allocate -H            # for help message; you may also want to set a reminder
            ws_list                   # check your workspaces

5) Point Hugging Face caches (in the shell script) to the shared filesystem to avoid filling $HOME:
   export HF_HOME=/data/horse/ws/<your_space>/hf_home
   export TRANSFORMERS_CACHE=/data/horse/ws/<your_space>/hf_cache
   export HF_DATASETS_CACHE=/data/horse/ws/<your_space>/hf_datasets

6) Authenticate to Hugging Face (only once per environment):
   huggingface-cli login --token <YOUR_HF_TOKEN>

7) Submit on Slurm (after editing 07-finetuning.sh):
   sbatch 07-finetuning.sh

   Helpful: squeue --me                #lists all your jobs (running, pending, etc.)
            watch squeue --me          #monitors your jobs live and updates every 2 seconds
            scancel <job_id>           #cancels the job with the given ID
"""

import argparse
import os
import json
import inspect
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
import evaluate
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LoRA fine-tune Qwen3-0.6B on a small instruction dataset, with baseline "
            "inference/eval before and after training."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="knkarthick/dialogsum",
        help="Public summarization dataset on Hugging Face Hub; small enough for a quick lab run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save LoRA adapters and tokenizer (must be on the shared filesystem).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for HF model/data caches on the shared filesystem (recommended).",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=600,
        help="Cap training samples to keep runtime under 1 hour. Set None to use full dataset.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
        help="Number of training epochs (use fractional values to cap runtime).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Per-GPU batch size. Lower if you hit OOM; increase if you have headroom.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="LoRA learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducibility seed.",
    )
    parser.add_argument(
        "--num_inference_samples",
        type=int,
        default=3,
        help="How many example prompts to generate for before/after comparisons.",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=50,
        help="How many eval samples to generate summaries for ROUGE scoring.",
    )
    return parser.parse_args()


@dataclass
class PromptExample:

    instruction: str
    input: str
    output: str

    def to_text(self) -> str:
        # Simple instruction-following template; keeps prompts short for quick training.
        return (
            f"### Instruction:\n{self.instruction}\n\n"
            f"### Input:\n{self.input or 'N/A'}\n\n"
            f"### Response:\n{self.output}"
        )


def build_prompt(example: Dict[str, str]) -> str:
    """Formats raw dataset rows into a plain text prompt."""
    # DialogSum adapter: dialogue -> input, summary -> output.
    if "dialogue" in example and "summary" in example:
        prompt = PromptExample(
            instruction="Summarize the following dialogue.",
            input=example.get("dialogue", ""),
            output=example.get("summary", ""),
        )
        return prompt.to_text()

    # Generic instruction-style fallback (for datasets with instruction/input/output columns).
    prompt = PromptExample(
        instruction=example.get("instruction", ""),
        input=example.get("input", ""),
        output=example.get("output", ""),
    )
    return prompt.to_text()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer first to set padding tokens.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model; 0.6B params is small enough for fp16/bf16 on a single GPU.
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    base_model.config.use_cache = False  # Needed when using gradient checkpointing or Trainer.

    # Load dataset and keep it tiny to finish within the lab slot.
    raw_dataset = load_dataset("csv", "data/train_dataset_mcq.csv",cache_dir=args.cache_dir)
    base_train = raw_dataset["train"]
    # Create a small eval split if one is missing.
    if "test" in raw_dataset:
        base_eval = raw_dataset["test"]
    else:
        split = base_train.train_test_split(test_size=0.05, seed=args.seed)
        base_train, base_eval = split["train"], split["test"]

    if args.max_train_samples is not None:
        base_train = base_train.select(range(min(args.max_train_samples, len(base_train))))

    # Save a few raw samples so students can inspect the dataset contents.
    sample_path = os.path.join(args.output_dir, "dataset_samples.json")
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(base_train.select(range(min(5, len(base_train)))).to_list(), f, indent=2)
    print(f"Wrote dataset samples to {sample_path}")

    # Map to prompts and tokenize.
    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        # Rebuild row-wise dicts because datasets.map provides column-wise batches.
        keys = list(batch.keys())
        batch_size = len(batch[keys[0]])
        rows = [{k: batch[k][i] for k in keys} for i in range(batch_size)]
        prompts = [build_prompt(row) for row in rows]
        tokenized = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_dataset = base_train.map(
        tokenize_batch,
        batched=True,
        remove_columns=base_train.column_names,
        desc="Tokenizing train",
    )
    eval_dataset = base_eval.map(
        tokenize_batch,
        batched=True,
        remove_columns=base_eval.column_names,
        desc="Tokenizing eval",
    )

    # Helper: build TrainingArguments with backward compatibility (filters unknown kwargs).
    def build_training_args(**kwargs) -> TrainingArguments:
        allowed = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return TrainingArguments(**filtered)

    # Helper: quick evaluation wrapper so we can compare before/after fine-tuning.
    def evaluate_model(model, tag: str) -> Dict[str, float]:
        eval_args = build_training_args(
            output_dir=os.path.join(args.output_dir, f"eval_{tag}"),
            per_device_eval_batch_size=args.per_device_train_batch_size,
            dataloader_drop_last=False,
            report_to="none",
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        eval_trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        metrics = eval_trainer.evaluate()
        if "eval_loss" in metrics:
            # Perplexity = exp(loss); lower means the model finds the text more likely.
            metrics["perplexity"] = float(torch.exp(torch.tensor(metrics["eval_loss"])))
        metrics_path = os.path.join(args.output_dir, f"{tag}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[{tag}] eval metrics: {metrics}")
        return metrics

    def count_trainable_parameters(model, tag: str) -> Dict[str, float]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        percent = 100.0 * trainable / total if total else 0.0
        print(f"[{tag}] trainable params: {trainable} / {total} ({percent:.2f}%)")
        counts = {"trainable": trainable, "total": total, "percent": percent}
        with open(os.path.join(args.output_dir, f"{tag}_params.json"), "w", encoding="utf-8") as f:
            json.dump(counts, f, indent=2)
        return counts

    rouge = evaluate.load("rouge")

    def generate_and_score(model, tag: str, num_samples: int) -> Dict[str, float]:
        """Generate summaries on a small subset and compute ROUGE."""
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        subset = base_eval.select(range(min(num_samples, len(base_eval))))
        generations = []
        preds, refs = [], []
        for idx, ex in enumerate(subset):
            prompt = build_prompt(ex)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            reference = ex.get("summary") or ex.get("output") or ""
            preds.append(decoded)
            refs.append(reference)
            generations.append(
                {"index": idx, "prompt": prompt, "prediction": decoded, "reference": reference}
            )
        metrics = rouge.compute(
            predictions=preds,
            references=refs,
            use_aggregator=True,
            use_stemmer=True,
        )
        gen_path = os.path.join(args.output_dir, f"{tag}_rouge_generations.json")
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(generations, f, indent=2)
        metrics_path = os.path.join(args.output_dir, f"{tag}_rouge.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[{tag}] ROUGE metrics: {metrics}")
        print(f"[{tag}] wrote generations to {gen_path}")
        return metrics

    # Helper: generate a few sample responses and save them for side-by-side comparison.
    def run_inference_samples(model, tag: str) -> None:
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        prompts = [build_prompt(base_eval[i]) for i in range(min(args.num_inference_samples, len(base_eval)))]
        generations = []
        with torch.no_grad():
            for idx, prompt in enumerate(prompts):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                generated = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                )
                gen_tokens = generated[0][inputs["input_ids"].shape[-1]:]
                decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                generations.append({"index": idx, "prompt": prompt, "generation": decoded})
        out_path = os.path.join(args.output_dir, f"{tag}_generations.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(generations, f, indent=2)
        print(f"[{tag}] wrote {len(generations)} generations to {out_path}")

    # Baseline: evaluate and generate with the base (unadapted) model.
    print("Running baseline inference/eval with the base model...")
    count_trainable_parameters(base_model, tag="base_model")
    run_inference_samples(base_model, tag="baseline")
    baseline_metrics = evaluate_model(base_model, tag="baseline")
    baseline_rouge = generate_and_score(base_model, tag="baseline", num_samples=args.num_eval_samples)

    # Configure LoRA; target_modules matches LLaMA/Qwen-style attention/MLP layers.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, lora_config)
    # Gradient checkpointing + LoRA needs inputs that require grad.
    if hasattr(lora_model, "enable_input_require_grads"):
        lora_model.enable_input_require_grads()
    count_trainable_parameters(lora_model, tag="lora_model")

    # Trainer handles shuffling and mixed precision for LoRA fine-tuning.
    training_args = build_training_args(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,  # Control runtime via epochs (can be fractional).
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=10,
        weight_decay=0.0,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),  
        fp16=False,
        report_to="none",
        dataloader_drop_last=True,
        gradient_checkpointing=False,  
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training finished. Adapter + tokenizer saved to {args.output_dir}")

    # Evaluate and generate after fine-tuning.
    print("Running post-finetune inference/eval...")
    finetune_metrics = trainer.evaluate()
    if "eval_loss" in finetune_metrics:
        finetune_metrics["perplexity"] = float(torch.exp(torch.tensor(finetune_metrics["eval_loss"])))
    finetune_path = os.path.join(args.output_dir, "finetuned_metrics.json")
    with open(finetune_path, "w", encoding="utf-8") as f:
        json.dump(finetune_metrics, f, indent=2)
    print(f"[finetuned] eval metrics: {finetune_metrics}")
    run_inference_samples(lora_model, tag="finetuned")
    generate_and_score(lora_model, tag="finetuned", num_samples=args.num_eval_samples)


if __name__ == "__main__":
    main()
