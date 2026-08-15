# Medical LLM Course Project

课程项目记录：围绕中文医疗大模型完成参数高效微调、偏好对齐、奖励模型实验和检索增强生成（RAG）。

## 内容

- `src/DPO`：基于 TRL 的 DPO 训练脚本；路径通过命令行参数传入，不包含模型权重或偏好数据集。
- `src/PPO`：奖励模型与 PPO 实验脚本。课程实验中 PPO 最终阶段因单卡显存限制未完成，保留脚本和问题记录。
- `src/RAG/rag.py`：LangChain + FAISS 的阿尔茨海默病文献问答实验代码；原始论文、索引和模型未上传。
- `src/benchmarks`：TruthfulQA、医疗问答和高诱导问题测试脚本。
- `results/`：课程实验导出的对比结果表，仅用于记录，不代表临床结论。

## 已完成实验

1. RTX 4090 24 GB 上完成 Llama-7B、2,000 steps LoRA 指令微调，产出 Adapter。
2. 使用 4-bit 量化完成 DPO 偏好对齐，并训练奖励模型；PPO 阶段定位到显存瓶颈。
3. 构建本地文献 RAG，完成 10 组问答对照及 TruthfulQA/安全性测试。

## 运行说明

仅建议在具备相应模型、数据集和显存的环境中运行。以 DPO 为例：

```bash
export MODEL_PATH=/path/to/base-model
export PEFT_PATH=/path/to/adapter
export DPO_DATASET_PATH=/path/to/dpo-dataset
bash src/DPO/run_dpo.sh
```

模型权重、训练数据、文献 PDF、FAISS 索引和缓存均不在仓库中。医疗问答结果仅作课程研究记录，不能替代专业诊疗意见。

## 致谢与归属

项目基于 [ChatMed](https://github.com/michaelwzh/ChatMed) 的公开工作进行课程实验和二次开发，相关上游代码、模型与数据集遵循各自许可证；本仓库只记录本人的实验脚本、配置整理和结果分析。
