import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

import rerank_client_sync

from model_definition import ModernBertWithActivationHeadModel
from model_support import batch_encode_attached


def sglang_reranker_fn(
    queries,
    top_k_passages,
    base_url="http://localhost:30000/v1",
    model="BAAI/bge-reranker-v2-m3",
):
    """
    Reranker function using SGLang reranker API.

    Args:
        queries: list of query strings
        top_k_passages: list of lists - top_k_passages[i] contains passages for queries[i]
        port: SGLang server port
        model: reranker model name

    Returns:
        list of argsorted indices for each query (relative to their passages)
    """

    config = rerank_client_sync.Configuration(host=base_url)

    with rerank_client_sync.ApiClient(config) as api_client:
        api = rerank_client_sync.DefaultApi(api_client)

        reranked_indices = []

        for query, passages in zip(queries, top_k_passages):
            request = rerank_client_sync.RerankRequest(
                query=query, documents=passages, model=model, return_documents=False
            )

            response = api.rerank(request)

            # Extract indices sorted by score (already sorted by API)
            indices = [result.index for result in response]

            reranked_indices.append(indices)

        return reranked_indices


def single_recall_at_k(sorted_indices, labels, k_values):
    """
    Compute recall@k for multiple k values efficiently.
    Recall@k = (relevant found in top k) / min(total_relevant, k)

    Args:
        sorted_indices: argsorted passage indices (could be subset if reranked)
        labels: binary labels for ALL passages (full length)
        k_values: list of k values to compute recall for

    Returns:
        dict: {k: recall@k} for each k
    """
    sorted_labels = labels[sorted_indices]
    # print("sorted_labels:", sorted_labels)
    relevant_cumsum = np.cumsum(sorted_labels)
    # print("relevant_cumsum:", relevant_cumsum)
    total_relevant = np.sum(labels)
    # print("total_relevant:", total_relevant)

    if total_relevant == 0:
        return {k: 0.0 for k in k_values}

    max_k = len(sorted_indices)
    if max(k_values) > max_k:
        raise ValueError(
            f"max(k_values)={max(k_values)} exceeds available sorted results ({max_k})"
        )

    recalls = {}
    for k in k_values:
        found = relevant_cumsum[k - 1]
        max_possible = min(total_relevant, k)
        recalls[k] = found / max_possible
        # print(f"k: {k}, found: {found}, max_possible: {max_possible}, recall@{k}={recalls[k]}")

    return recalls


def average_recall_at_k(
    queries,
    passages,
    similarity_matrix,
    labels_per_query,
    k_values,
    rerank_fn=None,
    rerank_k=None,
):
    """
    Compute recall@k across all queries with optional reranking.

    Args:
        queries: list of unique query strings
        passages: list of all passage strings
        similarity_matrix: [num_queries, num_passages] similarity scores
        labels_per_query: dict mapping query -> binary label array for all passages
        k_values: list of k values to compute recall for
        rerank_fn: optional reranker function(queries, top_k_passages) -> reranked_indices
        rerank_k: number of top passages to rerank (must be >= max(k_values))

    Returns:
        dict: {k: average_recall@k} across all queries
    """
    if rerank_fn is not None and rerank_k is None:
        raise ValueError("rerank_k must be specified when rerank_fn is provided")

    if rerank_fn is not None and rerank_k < max(k_values):
        raise ValueError(
            f"rerank_k ({rerank_k}) must be >= max(k_values) ({max(k_values)})"
        )

    all_recalls = {k: [] for k in k_values}

    for query_idx, query in enumerate(queries):
        # Step 1: Get initial ranking from similarity
        initial_sorted_indices = np.argsort(-similarity_matrix[query_idx])

        if rerank_fn is None:
            # Use full similarity ranking
            final_sorted_indices = initial_sorted_indices
        else:
            # Rerank top rerank_k
            top_k_indices = initial_sorted_indices[:rerank_k]
            top_k_passages = [passages[i] for i in top_k_indices]
            reranked_relative_list = rerank_fn([query], [top_k_passages])
            reranked_relative = reranked_relative_list[0]  # Get result for this query

            # Map back to absolute indices
            final_sorted_indices = top_k_indices[reranked_relative]

        # Compute recall for this query
        query_labels = labels_per_query[query]
        query_recalls = single_recall_at_k(final_sorted_indices, query_labels, k_values)

        # Accumulate results
        for k in k_values:
            all_recalls[k].append(query_recalls[k])

    # Average across all queries
    avg_recalls = {}
    for k in k_values:
        avg_recalls[k] = np.mean(all_recalls[k])

    return avg_recalls


def prepare_eval_data(eval_dataset):
    """
    Convert evaluation dataset triples into format needed for benchmarking.

    Args:
        eval_dataset: HF dataset with 'query', 'passage', 'label' columns

    Returns:
        tuple: (unique_queries, unique_passages, labels_per_query_dict)
    """
    # Get unique queries and passages
    unique_queries = list(set(eval_dataset["query"]))
    unique_passages = list(set(eval_dataset["passage"]))

    # Create passage -> index mapping
    passage_to_idx = {p: i for i, p in enumerate(unique_passages)}

    # Build labels per query
    labels_per_query = {}
    for query in unique_queries:
        labels_per_query[query] = np.zeros(len(unique_passages), dtype=int)

    # Fill in labels from triples
    for query, passage, label in zip(
        eval_dataset["query"], eval_dataset["passage"], eval_dataset["label"]
    ):
        passage_idx = passage_to_idx[passage]
        labels_per_query[query][passage_idx] = label

    return unique_queries, unique_passages, labels_per_query


def prepare_germanquad_for_benchmark(ds):
    """
    Convert HF Dataset (query/passage/label) to the format required by benchmark_model.
    """
    # Get unique queries and passages
    unique_queries = list(set(ds["query"]))
    unique_passages = list(set(ds["passage"]))

    # Map passage text → index
    passage_to_idx = {p: i for i, p in enumerate(unique_passages)}

    # Build labels_per_query dict
    labels_per_query = {
        q: np.zeros(len(unique_passages), dtype=int) for q in unique_queries
    }

    for q, p, l in zip(ds["query"], ds["passage"], ds["label"]):
        idx = passage_to_idx[p]
        labels_per_query[q][idx] = l

    return unique_queries, unique_passages, labels_per_query


def benchmark_model(
    model=None,
    tokenizer=None,
    eval_queries=None,
    eval_passages=None,
    labels_per_query=None,
    k_values=None,
    batch_size=32,
    rerank_fn=None,
    rerank_k=None,
):
    """
    Pure function: benchmark model with recall@k metrics.

    Args:
        model: student model to evaluate
        tokenizer: tokenizer for the student model
        eval_queries: list of unique query strings
        eval_passages: list of unique passage strings
        labels_per_query: dict mapping query -> binary label array for all passages
        k_values: list like [1, 5, 10, 20, 50, 100]
        batch_size: batch size for encoding
        rerank_fn: optional reranker function(queries, top_k_passages) -> reranked_indices
        rerank_k: rerank top-k (must be >= max(k_values))

    Returns:
        dict: {k: recall@k} averaged across all queries
    """
    # Encode queries and passages with current model
    with torch.no_grad():
        query_embeddings = batch_encode_attached(
            model, tokenizer, eval_queries, batch_size=batch_size
        )
        passage_embeddings = batch_encode_attached(
            model, tokenizer, eval_passages, batch_size=batch_size
        )

    # Compute similarity matrix
    similarity_matrix = query_embeddings @ passage_embeddings.T
    similarity_matrix = similarity_matrix.detach().cpu().numpy()

    # Compute recall@k metrics
    recalls = average_recall_at_k(
        eval_queries,
        eval_passages,
        similarity_matrix,
        labels_per_query,
        k_values,
        rerank_fn,
        rerank_k,
    )

    return recalls


def main():
    # -------------------------------
    # Step 0: Setup
    # -------------------------------
    base_url = os.environ.get("RERANKER_BASE_URL", "http://localhost:30000")

    # set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # use fast float32 computations
    torch.set_float32_matmul_precision("high")

    # -------------------------------
    # Step 1: Somewhat realistic multirelation dataset
    # -------------------------------
    queries = [
        "What causes rainbows in the sky?",
        "Who invented the telephone?",
        "Which planets have rings in our Solar System?",
        "What is the process of photosynthesis?",
        "Who painted the Mona Lisa?",
    ]

    all_passages = [
        "Rainbows are formed due to the refraction and dispersion of sunlight through water droplets in the atmosphere.",
        "Rainbows appear when sunlight interacts with raindrops after a storm.",
        "Clouds are collections of tiny water droplets or ice crystals.",
        "Alexander Graham Bell invented the first practical telephone in 1876.",
        "Antonio Meucci developed an early version of the telephone before Bell.",
        "The printing press was invented by Johannes Gutenberg in the 15th century.",
        "Saturn has a prominent system of rings made of ice and rock particles.",
        "Jupiter, Uranus, and Neptune also have ring systems.",
        "Earth does not have a ring system.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "Chlorophyll in leaves absorbs light energy to produce glucose and oxygen.",
        "Fossil fuels are formed from ancient organic matter.",
        "Leonardo da Vinci painted the Mona Lisa during the Italian Renaissance.",
        "The Last Supper is another famous painting by Leonardo da Vinci.",
        "Michelangelo sculpted David, not the Mona Lisa.",
    ]

    # Multi-relational relevance (indices of relevant passages)
    query_relevant_indices = {
        "What causes rainbows in the sky?": [0, 1],
        "Who invented the telephone?": [3, 4],
        "Which planets have rings in our Solar System?": [6, 7],
        "What is the process of photosynthesis?": [9, 10],
        "Who painted the Mona Lisa?": [12, 13],
    }

    # -------------------------------
    # Step 1a: Prepare top passages per query for reranker
    # -------------------------------
    top_k_passages_per_query = [
        [
            p
            for i, p in enumerate(all_passages)
            if i in query_relevant_indices[q] or i % 3 == 0
        ]
        for q in queries
    ]

    # Call reranker and display results
    reranked_results = sglang_reranker_fn(
        queries, top_k_passages_per_query, base_url=base_url
    )
    for q, r in zip(queries, reranked_results):
        print(f"\nQuery: {q}")
        print("Reranked indices:", r)
        print("Passages in order:")
        for idx in r:
            print(f"  - {top_k_passages_per_query[queries.index(q)][idx]}")

    # -------------------------------
    # Step 2: Convert to long-format dataset for benchmarking
    # -------------------------------
    eval_dataset = {"query": [], "passage": [], "label": []}
    for q in queries:
        rel_idx_list = query_relevant_indices[q]
        for i, p in enumerate(all_passages):
            eval_dataset["query"].append(q)
            eval_dataset["passage"].append(p)
            eval_dataset["label"].append(1 if i in rel_idx_list else 0)

    unique_queries, unique_passages, labels_per_query = prepare_eval_data(eval_dataset)
    k_values = [1, 5, 10]

    # -------------------------------
    # Step 3: Load evaluation model
    # -------------------------------
    eval_model_name = "models/ettin-encoder-17m"
    tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
    model = ModernBertWithActivationHeadModel.from_pretrained(eval_model_name)
    model.eval()
    model.to(device)

    # -------------------------------
    # Step 4: Benchmark w/o reranker
    # -------------------------------
    recalls = benchmark_model(
        model=model,
        tokenizer=tokenizer,
        eval_queries=unique_queries,
        eval_passages=unique_passages,
        labels_per_query=labels_per_query,
        k_values=k_values,
        batch_size=15,
    )

    # -------------------------------
    # Step 4a: Print results
    # -------------------------------
    print("\nBenchmark results w/o reranker (recall@k):")
    for k, r in recalls.items():
        print(f"Recall@{k}: {r:.4f}")

    # -------------------------------
    # Step 5: Benchmark with reranker
    # -------------------------------
    recalls = benchmark_model(
        model=model,
        tokenizer=tokenizer,
        eval_queries=unique_queries,
        eval_passages=unique_passages,
        labels_per_query=labels_per_query,
        k_values=k_values,
        batch_size=15,
        rerank_fn=lambda q, p: sglang_reranker_fn(q, p, base_url=base_url),
        rerank_k=15,
    )

    # -------------------------------
    # Step 5: Print results
    # -------------------------------
    print("\nBenchmark results with reranker (recall@k):")
    for k, r in recalls.items():
        print(f"Recall@{k}: {r:.4f}")


if __name__ == "__main__":
    main()
