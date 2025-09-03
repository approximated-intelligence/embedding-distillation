import torch


def batch_encode_detached(
    model,
    tokenizer,
    scorer,
    texts,
    batch_size=32,
    padding="longest",
    pad_to=16,
    truncation=True,
    max_length=8192,
):
    embeddings = []

    device = next(model.parameters()).device

    for i in range(0, len(texts), batch_size):
        in_batch = texts[i : i + batch_size]
        inputs = tokenizer(
            in_batch,
            return_tensors="pt",
            padding=padding,
            pad_to_multiple_of=pad_to,
            truncation=truncation,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
            hidden = hidden.detach()

        scores = scorer(hidden)
        mask = inputs["attention_mask"].unsqueeze(-1)
        batch_emb = (hidden * scores * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        embeddings.append(batch_emb)

    return torch.cat(embeddings, dim=0)


def batch_encode_attached(
    model,
    tokenizer,
    texts,
    batch_size=32,
    padding="longest",
    pad_to=16,
    truncation=True,
    max_length=8192,
):
    embeddings = []

    for i in range(0, len(texts), batch_size):
        in_batch = texts[i : i + batch_size]
        inputs = tokenizer(
            in_batch,
            return_tensors="pt",
            padding=padding,
            pad_to_multiple_of=pad_to,
            truncation=truncation,
            max_length=max_length,
        ).to(model.device)

        batch_emb = model(**inputs)["embeddings"]

        embeddings.append(batch_emb)

    return torch.cat(embeddings, dim=0)


def make_retriever(sim_matrix, queries, passages):
    # sim_matrix: [num_queries, num_passages] torch.Tensor
    # queries: list of query strings corresponding to rows of sim_matrix
    # passages: list of passage strings corresponding to columns

    query_to_idx = {q: i for i, q in enumerate(queries)}

    def retrieve(query, top_k=5):
        idx = query_to_idx[query]  # get row index
        sims = sim_matrix[idx]
        scores, indices = torch.topk(sims, k=top_k)
        top_passages = [passages[i] for i in indices]
        return top_passages, scores

    return retrieve


def batch_encode_bge_m3(
    model,
    tokenizer,
    texts,
    batch_size=32,
    padding="longest",
    pad_to=16,
    truncation=True,
    max_length=8192,
):
    embeddings = []
    device = model.model.device

    for i in range(0, len(texts), batch_size):
        in_batch = texts[i : i + batch_size]
        inputs = tokenizer(
            in_batch,
            return_tensors="pt",
            padding=padding,
            pad_to_multiple_of=pad_to,
            truncation=truncation,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            out = model(
                inputs,
                return_dense=False,
                return_sparse=True,
                return_sparse_embedding=True,
                return_colbert_vecs=False,
            )
            batch_emb = out["sparse_vecs"]
            embeddings.append(batch_emb)

    return torch.cat(embeddings, dim=0)
