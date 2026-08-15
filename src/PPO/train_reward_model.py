import argparse
import logging
import sys
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]="3" 
import torch

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model,TaskType
from trl import RewardTrainer, RewardConfig # <-- 添加 RewardConfig
from trl.trainer.utils import RewardDataCollatorWithPadding
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

def prepare_reward_data(examples, tokenizer):
    # 奖励模型通常是比较两个序列，所以需要 prompt + chosen 和 prompt + rejected
    formatted_prompts = []
    # 将 'prompt' 修改为 'input'，因为这是原始数据列名
    for i in range(len(examples['input'])): # <-- 修改这里
        system_message = examples['system'][i].strip() if examples['system'][i] else ""
        user_message = examples['input'][i].strip() # <-- 修改这里


        # 这里的 `system_message` 如果有内容，可以作为前置上下文或指令
        if system_message:
            # 示例：将系统指令放在问题之前，作为上下文的一部分
            formatted_prompt = f"<s>{system_message}\n\n问：{user_message}\n答："
            # 或者更像一个通用指令：
            # formatted_prompt = f"<s>请根据以下背景信息回答问题：\n{system_message}\n\n问题：{user_message}\n回答："
        else:
            # 简单的问答对
            formatted_prompt = f"<s>问：{user_message}\n答："

        formatted_prompts.append(formatted_prompt)

    # Concatenate prompt with chosen and rejected responses
    # Reward model takes a single string: prompt + response
    chosen_texts = [f"{p} {c.strip()}{tokenizer.eos_token}" for p, c in zip(formatted_prompts, examples['chosen'])]
    rejected_texts = [f"{p} {r.strip()}{tokenizer.eos_token}" for p, r in zip(formatted_prompts, examples['rejected'])]

    # Tokenize as pairs for comparison
    # RewardTrainer expects input_ids and attention_mask for each pair
    # And label indicating which is better (chosen=1, rejected=0)
    tokenized_chosen = tokenizer(chosen_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length") # Ensure padding="max_length"
    tokenized_rejected = tokenizer(rejected_texts, max_length=tokenizer.model_max_length, truncation=True, padding="max_length") # Ensure padding="max_length"
    
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }


def main():
    parser = argparse.ArgumentParser(description="Train a Reward Model for LLM alignment.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the base model for RM (e.g., Llama-7B-HF).")
    parser.add_argument("--tokenizer_name_or_path", type=str, help="Path to the tokenizer. Defaults to model_name_or_path.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Path to the DPO dataset.")
    parser.add_argument("--dataset_cache_dir", type=str, default="./rm_dataset_cache", help="Directory to cache the dataset.")
    parser.add_argument("--output_dir", type=str, default="./reward_model_output", help="Output directory for RM checkpoints.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulation steps.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for RM.")
    parser.add_argument("--fp16", action="store_true", help="Use FP16.")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank for RM.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha for RM.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout for RM.")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length.")

    args = parser.parse_args()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path,
        padding_side="right",
        legacy=False,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    # Set model_max_length if not set or too small
    if tokenizer.model_max_length > 1e10: # Default very large value often means it's not set
        tokenizer.model_max_length = args.max_length # <--- This is important
    
    # Load Base Model for Reward Model
    # Reward models are typically AutoModelForSequenceClassification
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=1, # Single output for reward score
        quantization_config=nf4_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # For sequence classification, adjust the final head if tokenizer pad_token was added
    if tokenizer.pad_token is not None and model.config.pad_token_id is None:
         model.config.pad_token_id = tokenizer.pad_token_id
         # resize_token_embeddings needed if new tokens were added.
         # For pad_token only, if it was already in vocab but just not set as config.pad_token_id, then no resize needed.
         # If it was added to vocab (e.g. from tokenizer.add_special_tokens), then resize_token_embeddings is critical.
         # model.resize_token_embeddings(len(tokenizer)) # Only if adding *new* tokens to vocab

    # Apply LoRA to Reward Model
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none", # Common for reward models
        task_type=TaskType.SEQ_CLS, # Task type for sequence classification
        target_modules=["q_proj", "v_proj"], # Or other relevant modules for RM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load Dataset
    raw_datasets = load_dataset("parquet", data_files=os.path.join(args.dataset_name, "train-00000-of-00001.parquet"), cache_dir=args.dataset_cache_dir)
    if isinstance(raw_datasets, DatasetDict):
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets.get("validation") or raw_datasets.get("test")
        if eval_dataset is None and len(train_dataset) > 1000:
            logger.info("No explicit validation set found. Creating a 10% split from training data.")
            split_datasets = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_datasets["train"]
            eval_dataset = split_datasets["test"]
    else:
        train_dataset = raw_datasets
        eval_dataset = None
    
    # Prepare dataset for RewardTrainer
    logger.info("Preparing reward dataset...")
    tokenized_train_dataset = train_dataset.map(
        lambda examples: prepare_reward_data(examples, tokenizer),
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=train_dataset.column_names, # Remove original columns
        desc="Tokenizing train dataset for RM",
    )
    if eval_dataset:
        tokenized_eval_dataset = eval_dataset.map(
            lambda examples: prepare_reward_data(examples, tokenizer),
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval dataset for RM",
        )
    else:
        tokenized_eval_dataset = None


    # TrainingArguments for Reward Model (using RewardConfig)
    training_args = RewardConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_strategy="steps" if tokenized_eval_dataset else "no",
        eval_steps=500 if tokenized_eval_dataset else None,
        report_to="none",
        center_rewards_coefficient=None,
    )
    # Create the data collator explicitly
    # Pass tokenizer and max_length here as required by RewardDataCollatorWithPadding
    data_collator = RewardDataCollatorWithPadding(
        tokenizer=tokenizer,
        #max_length=args.max_length, # Use the max_length from args
        # The `remove_unused_columns=False` warning is often handled by default
        # or within the collator's logic. If it persists, you might need to add it here.
        # However, it's often more robust if the Trainer or Collator manages it.
    )
    # Initialize RewardTrainer
    reward_trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # 👈 加上这行！非常重要
    )

    logger.info("*** Starting Reward Model Training ***")
    reward_trainer.train()

    logger.info("Saving final Reward Model...")
    reward_trainer.save_model(args.output_dir)
    logger.info("Reward Model training complete!")

if __name__ == "__main__":
    main()