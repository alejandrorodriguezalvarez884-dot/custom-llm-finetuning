"""
Fine-tune LittleLamb on your document dataset using LoRA (parameter-efficient
supervised fine-tuning). No GPU required — runs on CPU with float32 weights.

Usage:
    python src/trainer.py

Outputs LoRA adapter weights to outputs/model/.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "outputs" / "dataset" / "train.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "model"

MODEL_ID = "MultiverseComputingCAI/LittleLamb"

EPOCHS = int(os.getenv("FINETUNE_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("FINETUNE_BATCH_SIZE", "2"))
LORA_RANK = int(os.getenv("FINETUNE_LORA_RANK", "16"))
MAX_SEQ_LEN = int(os.getenv("FINETUNE_MAX_SEQ_LEN", "512"))
LEARNING_RATE = float(os.getenv("FINETUNE_LEARNING_RATE", "2e-4"))

# Qwen3 attention + MLP projection layers — standard LoRA targets
_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def main():
    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}.")
        print("Run dataset_builder.py first.")
        sys.exit(1)

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, TaskType
    from trl import SFTTrainer, SFTConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("Training on CPU — this may take a while. See README for time estimates.")

    # ------------------------------------------------------------------ model
    print(f"\nDownloading / loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,  # bitsandbytes 4-bit requires CUDA; use fp32 for CPU
        trust_remote_code=True,
    )

    # ----------------------------------------------------------------- dataset
    dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
    print(f"Training examples: {len(dataset)}")

    # -------------------------------------------------------------------- LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
        target_modules=_LORA_TARGET_MODULES,
        bias="none",
    )

    # ----------------------------------------------------------------- trainer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=False,
        fp16=False,
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        report_to="none",
        dataloader_pin_memory=False,  # not useful on CPU
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.model.print_trainable_parameters()

    # ----------------------------------------------------------------- train
    print("\nStarting fine-tuning...")
    trainer.train()

    # ------------------------------------------------------------------ save
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nLoRA adapters saved to {OUTPUT_DIR}")
    print("Run server.py to start the local inference API.")


if __name__ == "__main__":
    main()
