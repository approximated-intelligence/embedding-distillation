import rerank_client_sync

def sglang_reranker_fn(queries, top_k_passages, base_url="http://localhost:30000/v1", model="BAAI/bge-reranker-v2-m3"):
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
                query=query,
                documents=passages,
                model=model,
                return_documents=False
            )
            
            response = api.rerank(request)
            
            # Extract indices sorted by score (already sorted by API)
            indices = [result.index for result in response]

            reranked_indices.append(indices)
            
        return reranked_indices

def main():
    import os
    
    queries = [
        "What is the capital of France?",
        "Who developed the theory of relativity?"
    ]
    
    top_k_passages = [
        [
            "Paris is the capital and most populous city of France.",
            "Berlin is the capital of Germany.",
            "Madrid is the capital of Spain."
        ],
        [
            "Albert Einstein developed the theory of relativity.",
            "Isaac Newton formulated the laws of motion.",
            "Galileo made pioneering observations of the heavens."
        ]
    ]

    base_url = os.environ.get("RERANKER_BASE_URL", "http://localhost:30000")

    results = sglang_reranker_fn(queries, top_k_passages, base_url=base_url)

    for q, r in zip(queries, results):
        print(f"\nQuery: {q}")
        print("Reranked indices:", r)
        print("Passages in order:")
        for idx in r:
            print(f"  - {top_k_passages[queries.index(q)][idx]}")


if __name__ == "__main__":
    main()
