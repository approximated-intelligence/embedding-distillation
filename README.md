# Embedding Model Distillation

A PyTorch-based framework for distilling knowledge from large embedding models (teacher) into smaller, more efficient models (student) using learned attention pooling mechanisms and sparse representations.

<img alt="Model Distillation" src="https://repository-images.githubusercontent.com/1046829454/33d2cf57-ed58-44bc-8fa4-49b7c0704179" style="width: 100%; height: auto;" />

## Overview

This project implements knowledge distillation where smaller student models learn to mimic the embedding similarities produced by larger teacher models. The framework supports multiple pooling strategies and evaluation approaches for information retrieval tasks.

### Key Components

- **Teacher Model**: BGE-M3 (multilingual embedding model with sparse vector support)
- **Student Models**: ModernBert-based architectures with custom pooling heads
- **Distillation Target**: Similarity matrices between query-passage pairs using MSE loss
- **Pooling Strategies**: Dense attention-based pooling and sparse vocabulary-based pooling
- **Evaluation**: Comprehensive recall@k metrics with optional reranking

## Features

- **Knowledge Distillation**: Transfer similarity patterns from BGE-M3 to compact models
- **Dual Pooling Architectures**: 
  - `ModernBertWithActivationHeadModel`: Dense attention-weighted pooling
  - `ModernBertWithSparseHeadModel`: Sparse vocabulary-based representations
- **Multi-Dataset Support**: GermanQuAD, GermanDPR, and mMARCO datasets
- **Advanced Evaluation**: Recall@k metrics with SGLang reranker integration
- **Efficient Training**: Frozen backbone with trainable pooling heads only
- **Production Ready**: Proper model saving/loading and HuggingFace integration

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd embedding-distillation

# Install dependencies (requires uv)
uv install
```

## Architecture

### Dense Pooling Model (`ModernBertWithActivationHeadModel`)
```
Input → Tokenizer → ModernBert (frozen) → Activation Head → Attention-Weighted Pooling → Dense Embedding
```

### Sparse Pooling Model (`ModernBertWithSparseHeadModel`)
```
Input → Tokenizer → ModernBert (frozen) → Activation Head → Vocabulary Scatter → Sparse Embedding
```

### Training Pipeline
1. **Teacher Forward**: Generate sparse embeddings using BGE-M3
2. **Student Forward**: Generate embeddings using student model + pooling head
3. **Similarity Computation**: Calculate cosine similarities between query-passage pairs
4. **Loss Calculation**: MSE loss between Z-score normalized similarity matrices

## Quick Start

### Basic Training

```python
from trainer import AttachedPooledEmbedderTrainer
from model_definition import ModernBertWithActivationHeadModel
from data_loading import load_germandpr, passthrough_collator
from transformers import TrainingArguments
import torch

# Load models
student_model = ModernBertWithActivationHeadModel.from_pretrained("models/ettin-encoder-32m")
student_model.setup_for_training()

# Training setup
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=32,
    num_train_epochs=2,
    learning_rate=2e-4,
    save_strategy="steps",
    save_steps=128,
)

trainer = AttachedPooledEmbedderTrainer(
    model=student_model,
    model_tokenizer=student_tokenizer,
    bge_m3_model=teacher_model,
    bge_m3_tokenizer=teacher_tokenizer,
    train_dataset=load_germandpr(),
    data_collator=passthrough_collator,
    args=training_args,
)

trainer.train()
```

### Evaluation with Recall@k

```python
from metrics_support import benchmark_model, prepare_germanquad_for_benchmark
from data_loading import load_germanquad

# Prepare evaluation data
eval_dataset = load_germanquad(split="test")
unique_queries, unique_passages, labels_per_query = prepare_germanquad_for_benchmark(eval_dataset)

# Benchmark model
recalls = benchmark_model(
    model=student_model,
    tokenizer=student_tokenizer,
    eval_queries=unique_queries,
    eval_passages=unique_passages,
    labels_per_query=labels_per_query,
    k_values=[1, 5, 10, 20, 50],
    batch_size=32
)

print("Recall@k results:", recalls)
```

## Model Architectures

### Dense Attention Pooling

```python
from model_definition import ModernBertWithActivationHeadModel

# Dense model with attention-weighted pooling
model = ModernBertWithActivationHeadModel.from_pretrained("your-model-path")
model.setup_for_training()  # Freezes backbone, enables pooling head training
```

**Forward Process**:
- Compute attention scores for each token using linear head
- Apply ReLU activation to scores
- Weighted sum of hidden states using attention mask
- Normalize by sequence length

### Sparse Vocabulary Pooling

```python
from model_definition import ModernBertWithSparseHeadModel

# Sparse model with vocabulary-based representations
model = ModernBertWithSparseHeadModel.from_pretrained("your-model-path")
model.setup_for_training()
```

**Forward Process**:
- Compute activation scores for each token
- Scatter scores to vocabulary dimensions based on input token IDs
- Use `scatter_reduce` with "amax" to handle token repetitions
- Output: `[batch_size, vocab_size]` sparse embeddings

## Dataset Support

### Supported Datasets

| Dataset | Language | Type | Format |
|---------|----------|------|---------|
| GermanQuAD | German | QA | Question-context pairs (all positive) |
| GermanDPR | German | Retrieval | Query with positive/negative/hard negative contexts |
| mMARCO | Multilingual | Retrieval | Query-positive-negative triplets |

### Data Processing Pipeline

All datasets are normalized to:
```python
{
    "query": "What is machine learning?",
    "passage": "Machine learning is a subset of...",
    "label": 1  # 1 for relevant, 0 for irrelevant
}
```

### Loading Examples

```python
from data_loading import load_germandpr, load_germanquad, load_mmarco_multilang

# Single datasets
train_data = load_germandpr()
eval_data = load_germanquad(split="test")

# Multilingual combination
multilang_data = load_mmarco_multilang(languages=["english", "german", "french"])
```

## Training Strategies

### Detached Training (Legacy)

```python
from trainer import DetachedPooledEmbedderTrainer

# Uses separate activation head with detached gradients
trainer = DetachedPooledEmbedderTrainer(
    model=backbone_model,
    activation_head=separate_head,
    # ... other params
)
```

### Attached Training (Recommended)

```python
from trainer import AttachedPooledEmbedderTrainer

# Uses integrated model with built-in pooling head
trainer = AttachedPooledEmbedderTrainer(
    model=integrated_model,  # ModernBertWith*HeadModel
    # ... other params
)
```

## Evaluation Framework

### Recall@k Metrics

```python
from metrics_support import benchmark_model, sglang_reranker_fn

# Basic evaluation
recalls = benchmark_model(
    model=model,
    tokenizer=tokenizer,
    eval_queries=queries,
    eval_passages=passages,
    labels_per_query=labels,
    k_values=[1, 5, 10, 20, 50, 100]
)

# With reranking
recalls_reranked = benchmark_model(
    # ... same params ...
    rerank_fn=lambda q, p: sglang_reranker_fn(q, p, base_url="http://localhost:30000/v1"),
    rerank_k=50
)
```

### Training with Evaluation Callback

```python
from trainer import RecallEvaluationCallback

callback = RecallEvaluationCallback(
    model_tokenizer=tokenizer,
    eval_queries=eval_queries,
    eval_passages=eval_passages,
    labels_per_query=labels_per_query,
    k_values=[1, 5, 10, 20, 50],
    batch_size=32
)

trainer = AttachedPooledEmbedderTrainer(
    # ... trainer params ...
    callbacks=[callback]
)
```

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query_max_length` | 512 | Maximum tokens for queries |
| `passage_max_length` | 2048 | Maximum tokens for passages |
| `query_pad_to` | 16 | Padding multiple for queries |
| `passage_pad_to` | 64 | Padding multiple for passages |
| `batch_size` | 32 | Encoding batch size |
| `learning_rate` | 2e-4 | Learning rate for pooling head |
| `num_train_epochs` | 2 | Number of training epochs |

### Model Configuration

```python
# Teacher model setup
teacher = BGEM3FlagModel(
    "models/bge-m3",
    pooling_method="mean",
    # use_fp16=True  # Optional for speed
)

# Student model setup  
student = ModernBertWithActivationHeadModel.from_pretrained("models/ettin-encoder-32m")
student.setup_for_training()  # Freeze backbone, enable head training
```

## File Structure

```
├── data_loading.py         # Dataset loaders and preprocessors
├── model_definition.py     # Custom model architectures
├── model_support.py        # Encoding utilities for different model types
├── trainer.py             # Custom trainer classes and callbacks
├── metrics_support.py      # Evaluation metrics and benchmarking
├── main.py                # Example usage and testing
├── pyproject.toml         # Dependencies and project config
└── README.md              # This file
```

### Module Overview

- **`data_loading.py`**: Pure functions for dataset loading and batch processing
- **`model_definition.py`**: HuggingFace-compatible model classes with custom pooling
- **`model_support.py`**: Composable encoding functions for different model architectures
- **`trainer.py`**: Training logic with similarity-based distillation loss
- **`metrics_support.py`**: Evaluation pipeline with recall@k and reranking support

## Loss Function

Z-score normalized similarity matching:

```python
# Teacher similarities (frozen BGE-M3 sparse vectors)
sim_teacher = query_teacher_emb @ passage_teacher_emb.T
sim_teacher = (sim_teacher - sim_teacher.mean()) / sim_teacher.std()

# Student similarities (trainable pooling head)
sim_student = query_student_emb @ passage_student_emb.T  
sim_student = (sim_student - sim_student.mean()) / sim_student.std()

# MSE loss on normalized similarities
loss = F.mse_loss(sim_student, sim_teacher)
```

## Advanced Features

### SGLang Reranker Integration

```python
from metrics_support import sglang_reranker_fn

# Requires SGLang server running on specified port
reranked_indices = sglang_reranker_fn(
    queries=["query1", "query2"],
    top_k_passages=[["passage1", "passage2"], ["passage3", "passage4"]],
    base_url="http://localhost:30000/v1",
    model="BAAI/bge-reranker-v2-m3"
)
```

### Custom Model Extensions

```python
# Extend base model class
class CustomPoolingModel(ModernBertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.custom_head = nn.Sequential(
            nn.Linear(config.hidden_size, 2 * config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(2 * config.hidden_size, 1)
        )

    def setup_for_training(self):
        # Freeze backbone, enable custom head
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.custom_head.parameters():
            p.requires_grad = True
        return self
```

## Performance Optimization

1. **Mixed Precision**: Enable `use_fp16=True` for BGE-M3
2. **Padding Optimization**: Use `pad_to_multiple_of` for efficient tensor operations  
3. **Frozen Backbone**: Only train pooling heads to reduce memory usage
4. **Batch Processing**: Adjust batch size based on GPU memory
5. **Fast Math**: Use `torch.set_float32_matmul_precision("high")`

## Requirements

- Python 3.13+
- PyTorch 2.8+
- Transformers 4.20+
- Datasets <4.0
- FlagEmbedding 1.3.5+
- uv (package manager)

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{embedding-distillation,
  title={Embedding Model Distillation Framework with Learned Pooling},
  author={Christian Bahls},
  year={2025},
  url={https://github.com/approximated-intelligence/embedding-distillation}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [BGE-M3](https://github.com/FlagOpen/FlagEmbedding) for the teacher model and sparse embeddings
- [HuggingFace Transformers](https://github.com/huggingface/transformers) for the model framework
- [SGLang](https://github.com/sgl-project/sglang) for efficient reranker serving
- [ModernBert](https://huggingface.co/answer/ModernBert-base) for the efficient transformer backbone
