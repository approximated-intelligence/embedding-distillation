# Embedding Model Distillation

A PyTorch-based framework for distilling knowledge from large embedding models (teacher) into smaller, more efficient models (student) using learned attention pooling mechanisms.

<img alt="Model Distillation" src="https://repository-images.githubusercontent.com/1046829454/33d2cf57-ed58-44bc-8fa4-49b7c0704179" style="width: 100%; height: auto;" />

## Overview

This project implements a knowledge distillation approach where a smaller student model learns to mimic the embedding similarities produced by a larger teacher model. The framework uses:

- **Teacher Model**: BGE-M3 (a multilingual embedding model)
- **Student Model**: Ettin-Encoder-32M (a smaller transformer model)
- **Distillation Target**: Similarity matrices between query-passage pairs
- **Pooling Strategy**: Learned attention-based pooling using a trainable activation head

## Features

- **Knowledge Distillation**: Transfer knowledge from BGE-M3 to smaller models
- **Multi-Dataset Support**: GermanQuAD, GermanDPR, and mMARCO datasets
- **Learned Pooling**: Trainable attention mechanism for better representations
- **Multilingual**: Support for German and English retrieval tasks
- **Efficient Training**: Frozen backbone with only pooling head training
- **Similarity Matching**: MSE loss on normalized similarity matrices

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd embedding-distillation

# Install dependencies
uv install
```

## Quick Start

```python
from trainer import PooledEmbedderTrainer
from data_loading import load_germandpr, load_germanquad
import torch

# Basic usage
trainer = PooledEmbedderTrainer(
    model=student_model,
    model_tokenizer=student_tokenizer,
    activation_head=activation_head,
    bge_m3_model=teacher_model,
    bge_m3_tokenizer=teacher_tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # ... other parameters
)

trainer.train()
```

## Architecture

### Student Model Architecture
```
Input Text → Tokenizer → Transformer → Hidden States → Activation Head → Pooled Embedding
```

### Training Process
1. **Teacher Forward**: Generate embeddings using BGE-M3
2. **Student Forward**: Generate embeddings using student model + activation head
3. **Similarity Computation**: Calculate cosine similarities between query-passage pairs
4. **Loss Calculation**: MSE loss between normalized teacher and student similarities

## Dataset Support

### Supported Datasets

- **GermanQuAD**: German question-answering dataset
- **GermanDPR**: German dense passage retrieval dataset  
- **mMARCO**: Multilingual version of MS MARCO (English, German, French)

### Data Format

All datasets are converted to a unified format:
```python
{
    "query": "What is machine learning?",
    "passage": "Machine learning is a subset of artificial intelligence...",
    "label": 1  # 1 for positive pairs, 0 for negative pairs
}
```

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query_max_length` | 512 | Maximum tokens for queries |
| `passage_max_length` | 2048 | Maximum tokens for passages |
| `batch_size` | 32 | Batch size for encoding |
| `learning_rate` | 1e-3 | Learning rate for activation head |
| `num_train_epochs` | 3 | Number of training epochs |

### Model Configuration

```python
# Teacher model (BGE-M3)
teacher_model = BGEM3FlagModel(
    "BAAI/bge-m3",
    pooling_method="mean",
    # you can also enable: use_fp16=True
)

# Student model (any transformer)
student_model = AutoModel.from_pretrained("your-model-name")

# Learned pooling head
activation_head = torch.nn.Linear(hidden_dim, 1, bias=False)
```

## Training

### Run Training

```bash
uv run trainer.py
```

### Custom Training Script

```python
from trainer import PooledEmbedderTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=1e-3,
    logging_steps=100,
    save_strategy="epoch",
    eval_strategy="epoch",
)

trainer = PooledEmbedderTrainer(
    model=student_model,
    model_tokenizer=student_tokenizer,
    activation_head=activation_head,
    bge_m3_model=teacher_model,
    bge_m3_tokenizer=teacher_tokenizer,
    args=training_args,
    # ... other parameters
)

trainer.train()
```

## File Structure

```
├── data_loading.py      # Dataset loading and preprocessing
├── model_support.py     # Model utilities and batch encoding
├── trainer.py          # Custom trainer implementation
├── main.py             # Example usage and testing
└── README.md           # This file
```

### Key Components

- **`data_loading.py`**: Functions to load and preprocess datasets
- **`model_support.py`**: Utilities for batch encoding and retrieval
- **`trainer.py`**: Custom `PooledEmbedderTrainer` class extending HuggingFace Trainer
- **`main.py`**: Example script showing usage

## Loss Function

The framework uses Mean Squared Error (MSE) loss between normalized similarity matrices:

```python
# Teacher similarities (frozen)
sim_teacher = query_teacher_emb @ passage_teacher_emb.T
sim_teacher = (sim_teacher - sim_teacher.mean()) / sim_teacher.std()

# Student similarities (trainable)
sim_student = query_student_emb @ passage_student_emb.T  
sim_student = (sim_student - sim_student.mean()) / sim_student.std()

# MSE loss
loss = F.mse_loss(sim_student, sim_teacher)
```

## Advanced Usage

### Custom Activation Head

```python
# Simple linear head (default)
activation_head = torch.nn.Linear(hidden_dim, 1, bias=False)

# Multi-layer head with dropout
activation_head = torch.nn.Sequential(
    torch.nn.Linear(hidden_dim, 2 * hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.1),
    torch.nn.Linear(2 * hidden_dim, 1),
)
```

### Multi-language Training

```python
from data_loading import load_mmarco_multilang

# Load multilingual data
train_dataset = load_mmarco_multilang(
    languages=["english", "german", "french"],
    split="train"
)
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.20+
- Datasets 2.0+ (<4.0)
- FlagEmbedding

## Performance Tips

1. **Use FP16**: Enable `use_fp16=True` for BGE-M3 to speed up inference
2. **Batch Size**: Adjust batch size based on GPU memory
3. **Frozen Backbone**: Only the activation head is trained, keeping memory usage low

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{embedding-distillation,
  title={Embedding Model Distillation Framework},
  author={Christian Bahls},
  year={2025},
  url={https://github.com/approximated-intelligence/embedding-distillation}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [BGE-M3](https://github.com/FlagOpen/FlagEmbedding) for the teacher model
- [HuggingFace Transformers](https://github.com/huggingface/transformers) for the framework
- [Datasets](https://github.com/huggingface/datasets) for dataset handling
