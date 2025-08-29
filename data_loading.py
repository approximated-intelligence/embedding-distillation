import torch

from FlagEmbedding import BGEM3FlagModel

from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import Trainer
from transformers import TrainingArguments
from datasets import load_dataset
from datasets import concatenate_datasets
from datasets import Dataset


def batch_expand_germanquad(batch):
    # Each question-context pair is positive → label = 1
    expanded = [(q, ctx, 1) for q, ctx in zip(batch["question"], batch["context"])]

    queries, passages, labels = zip(*expanded) if expanded else ([], [], [])

    return {"query": list(queries), "passage": list(passages), "label": list(labels)}


def batch_expand_germandpr(batch):
    expanded = [
        (q, text, label)
        for q, pos_ctx, neg_ctx, hard_ctx in zip(
            batch["question"],
            batch["positive_ctxs"],
            batch["negative_ctxs"],
            batch["hard_negative_ctxs"],
        )
        for text, label in (
            [(t, 1) for t in pos_ctx.get("text", [])]
            + [(t, 0) for t in neg_ctx.get("text", [])]
            + [(t, 0) for t in hard_ctx.get("text", [])]
        )
    ]

    queries, passages, labels = zip(*expanded) if expanded else ([], [], [])

    return {"query": list(queries), "passage": list(passages), "label": list(labels)}


def batch_expand_mmarco(batch):
    # Generate all (query, passage, label) tuples in one pass
    expanded = [
        (q, p, label)
        for q, pos, neg in zip(batch["query"], batch["positive"], batch["negative"])
        for p, label in [(pos, 1), (neg, 0)]
    ]

    # Unzip into separate lists; handle empty batch
    queries, passages, labels = zip(*expanded) if expanded else ([], [], [])

    return {"query": list(queries), "passage": list(passages), "label": list(labels)}


def load_germanquad(split="train"):
    """
    Load GermanQuAD dataset and expand into query-passage-label format.
    """
    ds = load_dataset("deepset/germanquad", split=split, trust_remote_code=True)
    # print(ds[0])
    return ds.map(batch_expand_germanquad, batched=True, remove_columns=ds.column_names)


def load_germandpr(split="train"):
    """
    Load German DPR dataset and expand into query-passage-label format.
    """
    ds = load_dataset("deepset/germandpr", split=split, trust_remote_code=True)
    # print(ds[0])
    return ds.map(batch_expand_germandpr, batched=True, remove_columns=ds.column_names)


def load_mmarco(split="train", lang="english"):
    """
    Load mMARCO dataset and expand into query-passage-label format.
    """
    ds = load_dataset("unicamp-dl/mmarco", lang, split=split, trust_remote_code=True)
    # print(ds[0])
    return ds.map(batch_expand_mmarco, batched=True, remove_columns=ds.column_names)


mmarco_languages = ["english", "german", "french"]


def load_mmarco_multilang(languages=mmarco_languages, split="train"):
    datasets_per_lang = [load_mmarco(split=split, lang=lang) for lang in languages]
    return concatenate_datasets(datasets_per_lang)


def make_cross_product_dataset(queries, passages):
    data = [
        {"query": q, "passage": p, "label": 1 if i == j else 0}
        for i, q in enumerate(queries)
        for j, p in enumerate(passages)
    ]
    return Dataset.from_list(data)


def passthrough_collator(features):
    batch = {}
    for k in features[0].keys():
        batch[k] = [f[k] for f in features]
    return batch


