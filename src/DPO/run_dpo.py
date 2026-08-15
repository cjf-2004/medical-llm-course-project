import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]="3" 
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict

# 导入 trl 库
from trl import DPOConfig, DPOTrainer

# 您可以从原来的脚本中复制 ModelArguments 和 DataTrainingArguments
# 或者简化它们，只保留 DPO 训练需要的参数。
# 为了保持简洁，这里只保留 DPO 相关的关键参数。
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "The model checkpoint for DPO."})
    tokenizer_name_or_path: Optional[str] = field(default=None, metadata={"help": "The tokenizer path."})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the preference dataset (JSONL)."})
    dataset_cache_dir: Optional[str] = field(default=None, metadata={"help": "Where to store the cached data."})
    max_seq_length: Optional[int] = field(
        default=256, metadata={"help": "The maximum sequence length for prompts and responses."}
    )

@dataclass
class DPOTrainingArguments(TrainingArguments):
    # DPO 训练特有的参数
    beta: Optional[float] = field(default=0.1, metadata={"help": "The beta parameter for DPO loss."})
    loss_type: Optional[str] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss (e.g., 'sigmoid', 'hinge')."},
    )
    # LoRA 相关参数 (如果继续使用 LoRA)
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default='embed_tokens,lm_head') # DPO中通常不保存这些
    peft_path : Optional[str] = field(default=None) # 如果从已有的 LoRA checkpoint 开始 DPO

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DPOTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, dpo_training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, dpo_training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)],)

    logger.warning(
        f"Process rank: {dpo_training_args.local_rank}, device: {dpo_training_args.device}, n_gpu: {dpo_training_args.n_gpu}"
        + f"distributed training: {bool(dpo_training_args.local_rank != -1)}, 16-bits training: {dpo_training_args.fp16}"
    )
    set_seed(dpo_training_args.seed)

    # 1. 加载 Tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path or model_args.model_name_or_path,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, # Set to True for 4-bit quantization
        bnb_4bit_quant_type="nf4", # Recommended for 4-bit training
        bnb_4bit_compute_dtype=torch.float16, # Use float16 for computation if fp16 is enabled
        bnb_4bit_use_double_quant=True, # Double quantization for slightly better performance
    )
 
    # 2. 加载 SFT 模型 (作为 Policy Model 和 Reference Model)
    # The torch_dtype argument is now largely handled by BitsAndBytesConfig
    # You can remove or comment out `torch_dtype=torch_dtype` from from_pretrained calls
    # as `bnb_4bit_compute_dtype` in the config dictates the compute dtype.

    base_model_path = model_args.model_name_or_path # 确保这里是Llama基础模型路径
    
    # Load Policy Model with quantization
    model = LlamaForCausalLM.from_pretrained(
        base_model_path, # 这里加载的是 Llama 基础模型
        # torch_dtype=torch_dtype, # REMOVED: Handled by quantization_config
        quantization_config=quantization_config, # ADDED: Apply quantization
        device_map={"": dpo_training_args.local_rank} if dpo_training_args.local_rank != -1 else "auto",
    )

    # Load Reference Model with quantization
    ref_model = LlamaForCausalLM.from_pretrained(
        base_model_path, # 这里加载的也是 Llama 基础模型
        # torch_dtype=torch_dtype, # REMOVED: Handled by quantization_config
        quantization_config=quantization_config, # ADDED: Apply quantization
        device_map={"": dpo_training_args.local_rank} if dpo_training_args.local_rank != -1 else "auto",
    )

    # 3. 应用 LoRA adapter
    # 如果你之前 SFT 训练后保存的是 LoRA adapter，那么 dpo_training_args.peft_path
    # 应该指向你 SFT 训练结果的目录 (例如 "./sft_model_output")
    if dpo_training_args.peft_path is not None:
        logger.info(f"Loading PEFT adapter for policy model from {dpo_training_args.peft_path}")
        model = PeftModel.from_pretrained(model, dpo_training_args.peft_path, is_trainable=True) # 设置 is_trainable=True
        
        logger.info(f"Loading PEFT adapter for reference model from {dpo_training_args.peft_path}")
        ref_model = PeftModel.from_pretrained(ref_model, dpo_training_args.peft_path, is_trainable=False) # Ref model 不可训练
    else:
        logger.info("No PEFT adapter path provided. Initializing new PEFT model for DPO.")
        # 如果没有提供 pre-trained PEFT path，则根据 LoraConfig 重新初始化 LoRA
        target_modules = dpo_training_args.trainable.split(',')
        lora_rank = dpo_training_args.lora_rank
        lora_dropout = dpo_training_args.lora_dropout
        lora_alpha = dpo_training_args.lora_alpha
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        # 对于 ref_model，如果你没有提供 pre-trained adapter，且希望它保持不变，
        # 最好确保它不加载任何 LoRA 或加载后将其冻结
        # DPOTrainer 通常会处理 ref_model 的可训练性，但这里明确一下更好
        ref_model = get_peft_model(ref_model, peft_config) # 即使加载了，DPOTrainer也会冻结
        ref_model.requires_grad_(False) # 确保 ref_model 完全冻结
    
    # ADD THIS LINE HERE:
    # Ensure use_cache is False when gradient_checkpointing is True
    # This is a common requirement for gradient checkpointing to work with generate() in TRL
    # model.config.use_cache = False 
    # ref_model.config.use_cache = False # Also set for ref_model, just in case
    # # 打印可训练参数
    model.print_trainable_parameters()
    
    # 重要：DPOTrainer 会处理 reference model 的 LoRA 状态，确保它不被训练。
    # 理论上，ref_model 不应该有可训练的 LoRA 参数。
    # 如果您直接加载 LoRA adapter 到 ref_model，请确保其参数被冻结。
    # DPOTrainer 会在内部处理 ref_model 的 requires_grad 属性。

    # 4. 准备 DPO 数据集
    # 核心修改在这里：加载本地文件
    # 假设你的数据集解压后，训练数据文件在 `your_local_data_folder/train-00000-of-00001.parquet`
    # 请根据你实际解压后的文件路径来调整 `local_data_file_path`
    
    # 确保 data_args.dataset_name 现在指向你的本地文件夹
    # 例如，如果你解压到 `./data/distilabel-intel-orca-dpo-pairs/` 
    # 并且里面有 `train-00000-of-00001.parquet`
    
    # 示例1: 如果数据集只有一个文件 (例如 train.parquet 或 train.jsonl)
    # 你需要知道解压后的具体文件路径
    local_data_file_path = os.path.join(data_args.dataset_name, "train-00000-of-00001.parquet") # 假设是 parquet 文件
    
    # 注意: `distilabel-intel-orca-dpo-pairs` 通常只有一个 `train` split
    # 并且数据是 parquet 格式。
    
    logger.info(f"Loading local dataset from: {local_data_file_path}")
    
    # 指定文件类型，并传入具体的文件路径
    # 如果文件是 .jsonl 格式，type_argument 就改为 "json"
    # 如果是 .csv 格式，就改为 "csv"
    raw_datasets = load_dataset(
        "parquet",  # 指定文件类型，例如 "parquet", "json", "csv"
        data_files=local_data_file_path,
        cache_dir=data_args.dataset_cache_dir, # 继续使用缓存目录，加载本地文件也会有缓存
    )

    raw_datasets = raw_datasets.filter(
    lambda r: 
        r["status"] != "tie" and 
        r["chosen_score"] >= 8 and 
        not r["in_gsm8k_train"]
    )

    # DPOTrainer 期望数据集是 dict of datasets (例如 DatasetDict({'train': ...}))
    # 如果 load_dataset 直接返回一个 Dataset 对象（通常是针对单文件），
    # 我们需要将其包装成 DatasetDict。
    if isinstance(raw_datasets, DatasetDict):
        train_dataset = raw_datasets["train"]
    else: # 如果只有一个 split，load_dataset 会直接返回 Dataset 对象
        train_dataset = raw_datasets
        # 确保数据集有 DPO 所需的 'prompt', 'chosen', 'rejected' 列
        # 如果列名不一致，需要在这里进行重命名
        # 例如：
        # train_dataset = train_dataset.rename_columns({
        #     'instruction': 'prompt',
        #     'output_chosen': 'chosen',
        #     'output_rejected': 'rejected'
        # }) 
        # 但 'distilabel-intel-orca-dpo-pairs' 应该已经符合要求了。
    try:
    # Only rename 'input' to 'prompt', 'chosen' and 'rejected' are already correct.
        train_dataset = train_dataset.rename_columns({"input": "prompt"})
        logger.info("Successfully renamed 'input' column to 'prompt'.")
    except KeyError as e:
        logger.error(f"Error renaming columns. Make sure your dataset has the 'input' column. Missing key: {e}")
        logger.info(f"Available columns are: {train_dataset.column_names}")
        sys.exit(1) # Exit if the column is not found

    logger.info(f"Dataset columns after renaming: {train_dataset.column_names}")
    
    # 5. 初始化 DPOTrainer
    dpo_config = DPOConfig(
        output_dir=dpo_training_args.output_dir,
        per_device_train_batch_size=dpo_training_args.per_device_train_batch_size,
        gradient_accumulation_steps=dpo_training_args.gradient_accumulation_steps,
        learning_rate=dpo_training_args.learning_rate,
        num_train_epochs=dpo_training_args.num_train_epochs,
        max_length=data_args.max_seq_length, # 最大序列长度，用于截断
        max_prompt_length=data_args.max_seq_length // 2, # 通常 prompt 长度限制为总长度的一半
        #evaluation_strategy=dpo_training_args.evaluation_strategy,
        save_strategy=dpo_training_args.save_strategy,
        logging_steps=dpo_training_args.logging_steps,
        # DPO 特有参数
        beta=dpo_training_args.beta,
        loss_type=dpo_training_args.loss_type,
        # 其他 TrainingArguments 相关的参数
        fp16=dpo_training_args.fp16,
        bf16=dpo_training_args.bf16,
        report_to=dpo_training_args.report_to,
        remove_unused_columns=False, # DPOTrainer 会处理列，所以不要删除未使用的列
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        # eval_dataset=eval_dataset, # 如果有验证集
        # 这里不需要提供 compute_metrics，因为 DPO 损失是其主要指标
    )

    # 6. 开始 DPO 训练
    logger.info("*** Starting DPO training ***")
    dpo_trainer.train()

    # 7. 保存 DPO 后的模型
    dpo_trainer.save_model(dpo_training_args.output_dir)
    tokenizer.save_pretrained(dpo_training_args.output_dir)
    logger.info(f"DPO model saved to {dpo_training_args.output_dir}")

if __name__ == "__main__":
    main()