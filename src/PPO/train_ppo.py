import argparse
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]="3" 
import torch
import warnings
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model
from trl import PPOConfig, PPOTrainer # Ensure trl is 0.7.0
from transformers import AutoModelForSequenceClassification # For Reward Model


# Suppress specific warnings if desired
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training script.")
    # Model and Tokenizer arguments
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base model name or path.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Tokenizer name or path (defaults to model_name_or_path if not provided).")
    parser.add_argument("--peft_path", type=str, default=None, help="Optional path to an existing PEFT adapter for LoRA initialization.")
    parser.add_argument("--reward_model_path", type=str, default="./reward_model_output", help="Path to the trained reward model.")

    # Data arguments
    parser.add_argument("--dataset_path", type=str, default="./train.parquet", help="Path to the training dataset (parquet file).")
    parser.add_argument("--num_proc", type=int, default=os.cpu_count(), help="Number of processes for tokenization.")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum length for the tokenized prompt.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum total sequence length for generation.")

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj", help="Comma-separated list of module names to apply LoRA to.")

    # PPO arguments
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for PPO training.")
    parser.add_argument("--kl_penalty", type=float, default=0.01, help="Initial KL penalty coefficient.")
    parser.add_argument("--adap_kl_ctrl", type=bool, default=True, help="Whether to use adaptive KL control.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--batch_size", type=int, default=1, help="Total batch size for PPO rollout.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    # Add ppo_epochs back to argparse as it's a PPOConfig param in 0.7.0
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO epochs per rollout.")


    # Output and logging
    parser.add_argument("--output_dir", type=str, default="./ppo_model_output", help="Output directory for saving the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval for PPO steps.")

    return parser.parse_args()


# Helper function to prepare data for PPO
# Helper function to prepare data for PPO
def prepare_ppo_data(examples, tokenizer, max_prompt_length):
    formatted_prompts = []
    # No need to gather original_systems and original_inputs if you're not returning them
    # and they are embedded within formatted_prompts.
    # original_systems = []
    # original_inputs = []

    for i in range(len(examples['input'])):
        system_message = examples['system'][i].strip() if examples['system'][i] else ""
        user_message = examples['input'][i].strip()

        # original_systems.append(system_message) # NO LONGER NEEDED TO APPEND HERE
        # original_inputs.append(user_message)   # NO LONGER NEEDED TO APPEND HERE

        # Llama 1 compatible prompt formatting
        if system_message:
            formatted_prompt = f"<s>{system_message}\n\n问：{user_message}\n答："
        else:
            formatted_prompt = f"<s>问：{user_message}\n答："
        formatted_prompts.append(formatted_prompt)

    tokenized_prompts = tokenizer(
        formatted_prompts,
        max_length=max_prompt_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
        # IMPORTANT: Keep return_tensors removed.
    )

    return {
        "query": formatted_prompts, # This is the full formatted prompt string
        "input_ids": tokenized_prompts["input_ids"],
        "attention_mask": tokenized_prompts["attention_mask"],
        # REMOVE THESE LINES:
        # "original_system": original_systems,
        # "original_input": original_inputs,
    }


def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Using device: {device}")

    # Load Tokenizer
    tokenizer_path = args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left", legacy=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    tokenizer.model_max_length = args.max_length # Ensure tokenizer's max length is set

    # Quantization config for base model
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load Policy Model
    print("INFO: Loading Policy Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=nf4_config,
        device_map="auto", # Use accelerate for device mapping
        torch_dtype=torch.bfloat16,
    )

    if args.peft_path:
        print(f"INFO: Loading PEFT adapter for policy model from {args.peft_path}")
        model = PeftModel.from_pretrained(base_model, args.peft_path, is_trainable=True)
    else:
        print("INFO: Initializing new PEFT adapter for policy model.")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(','), # Common target modules for Llama
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False # Required for gradient checkpointing

    # Load Reference Model (same as policy model, but frozen)
    print("INFO: Loading Reference Model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=nf4_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # ref_model.to(device) # Device mapping handled by accelerate


    # Load Reward Model
    print(f"INFO: Loading Reward Model from: {args.reward_model_path}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        num_labels=1,
        quantization_config=nf4_config, # <--- ADD THIS LINE!
        device_map="auto", # Keep this to allow accelerate to manage
        torch_dtype=torch.bfloat16, # Keep this
    )
    reward_model.eval() # Ensure reward model is in evaluation mode


    # Load and prepare dataset
    print("INFO: Loading dataset...")
    if os.path.isdir(args.dataset_path):
        raw_dataset = load_dataset("parquet", data_dir=args.dataset_path)
    else:
        raw_dataset = load_dataset("parquet", data_files=args.dataset_path)

    if isinstance(raw_dataset, DatasetDict):
        train_dataset = raw_dataset['train']
    elif isinstance(raw_dataset, Dataset):
        train_dataset = raw_dataset
    else:
        raise TypeError(f"Unsupported dataset format: {type(raw_dataset)}")

    print(f"INFO: Raw training dataset loaded with {len(train_dataset)} examples.")

    print(f"INFO: Tokenizing train dataset for PPO (num_proc={args.num_proc})...")
    print("INFO: Preparing PPO dataset...")
    ppo_dataset = train_dataset.map(
        lambda examples: prepare_ppo_data(examples, tokenizer, args.max_prompt_length),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=train_dataset.column_names # Adjust this list based on your original dataset columns
    )

    # Crucial: Set the format to PyTorch AFTER mapping
    ppo_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    #print(f"INFO: PPO dataset prepared. Example entry: {ppo_dataset[0]}")


    # Initialize PPOConfig for trl==0.19.1 based on your provided parameters
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        mini_batch_size=args.per_device_train_batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        batch_size=args.batch_size, 
        report_to="tensorboard",  # Reconfirmed: 'report_to' for logging backend
        logging_dir=os.path.join(args.output_dir, "logs"), # Reconfirmed: 'logging_dir' for log path
        
        # New parameters based on your list:
        num_ppo_epochs=args.ppo_epochs, # This needs to be added back to argparse if it's not already -- let's assume you'll add it.
                                        # Or rename args.num_train_epochs to args.num_ppo_epochs in parse_args()
        whiten_rewards=True, # Assuming you want to enable this, or make it an argparse arg
        kl_coef=args.kl_penalty, # This replaces target_kl, init_kl_coef, adap_kl_ctrl
        kl_estimator="k1", # Default value, or make it an argparse arg
        cliprange=0.2, # Default value, or make it an argparse arg
        vf_coef=0.1, # Default value, or make it an argparse arg
        cliprange_value=0.2, # Default value, or make it an argparse arg
        gamma=1.0, # Default value, or make it an argparse arg
        lam=0.95, # Default value, or make it an argparse arg
        ds3_gather_for_generation=True, # Default value, or make it an argparse arg
    )

    # Define generation_kwargs to pass to ppo_trainer.generate()
    generation_kwargs = {
        "max_new_tokens": args.max_length - args.max_prompt_length,
        "do_sample": True,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 1.0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }



    print("INFO: Initializing PPOTrainer...")
    ppo_trainer = PPOTrainer(
        args=ppo_config,       # Your PPOConfig object
        processing_class=tokenizer, # Your tokenizer
        model=model,           # Your policy model (AutoModelForCausalLM with PEFT)
        ref_model=ref_model,   # Your separate, frozen reference model
        reward_model=reward_model, # Your loaded reward model
        train_dataset=ppo_dataset, # Your prepared PPO dataset
        value_model=model,     # Assuming your policy model will serve as the value model
        # data_collator is optional, defaults to DataCollatorWithPadding(processing_class)
        # eval_dataset, optimizers, callbacks, peft_config are optional
    )

    # PPO Training Loop
    print("INFO: Starting PPO training...")
    # PPO Training Loop
    print("INFO: Starting PPO training...")
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(ppo_trainer.dataloader):
            # Explicitly ensure query_tensors and attention_mask are Tensors
            # This is the most crucial part for this specific error
            query_tensors = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Ensure they are LongTensors as expected by embedding layer
            if query_tensors.dtype != torch.long:
                query_tensors = query_tensors.long()
            if attention_mask.dtype != torch.long: # Attention mask can be bool or float, but long is safe
                attention_mask = attention_mask.long()

            # Store the original text queries (the formatted prompts) for reward model text reconstruction
            # This 'query' column now contains your formatted_prompt strings directly
            queries_text = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
            
            # REMOVE THESE LINES, as 'original_system' and 'original_input' are no longer in 'batch'
            # original_systems = batch["original_system"]
            # original_inputs = batch["original_input"]

            # Generate response from policy model
            response_tensors = model.generate( # <--- Change this line
                query_tensors,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

            # Detokenize to pass to reward model, being careful with special tokens
            # responses are the *generated part only*
            responses = tokenizer.batch_decode(response_tensors[:, query_tensors.shape[1]:], skip_special_tokens=True)

            # Prepare inputs for reward model (prompt + response)
            reward_texts = []
            for i in range(len(responses)):
                # Use the full formatted prompt directly from queries_text
                full_prompt_for_rm = queries_text[i] # This already contains system and user message in Llama format
                response_msg = responses[i].strip()
                
                # The full text for the reward model is the formatted prompt + the generated response + EOS token
                reward_texts.append(f"{full_prompt_for_rm}{response_msg}{tokenizer.eos_token}")

            # Compute rewards
            with torch.no_grad():
                tokenized_reward_inputs = tokenizer(
                    reward_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length
                )
                # Extract input_ids and attention_mask and move them to device
                rm_input_ids = tokenized_reward_inputs["input_ids"].to(device)
                rm_attention_mask = tokenized_reward_inputs["attention_mask"].to(device)
                
                # Pass as explicit keyword arguments
                rewards = reward_model(input_ids=rm_input_ids, attention_mask=rm_attention_mask).logits.squeeze(1)
            # PPO update step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            if (step + 1) % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.num_train_epochs}, Step {step+1}: "
                      f"Reward: {stats['rewards/mean']:.4f}, "
                      f"KL: {stats['objective/kl']:.4f}, "
                      f"Policy Loss: {stats['losses/policy/loss']:.4f}, "
                      f"Value Loss: {stats['losses/value/loss']:.4f}")
                
    # Save the trained model
    output_model_path = os.path.join(args.output_dir, "final_ppo_model")
    ppo_trainer.save_pretrained(output_model_path)
    print(f"INFO: PPO trained model saved to {output_model_path}")

if __name__ == "__main__":
    main()
