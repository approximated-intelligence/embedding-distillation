# RerankRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**query** | **str** | The query text to rank documents against | 
**documents** | **List[str]** | List of documents to be ranked | 
**model** | **str** | Model to use for reranking | [optional] [default to 'BAAI/bge-reranker-v2-m3']
**top_k** | **int** | Maximum number of documents to return | [optional] 
**return_documents** | **bool** | Whether to return documents in addition to scores | [optional] [default to True]
**rid** | [**RerankRequestRid**](RerankRequestRid.md) |  | [optional] 
**user** | **str** | User identifier | [optional] 

## Example

```python
from rerank_client_sync.models.rerank_request import RerankRequest

# TODO update the JSON string below
json = "{}"
# create an instance of RerankRequest from a JSON string
rerank_request_instance = RerankRequest.from_json(json)
# print the JSON string representation of the object
print(RerankRequest.to_json())

# convert the object into a dict
rerank_request_dict = rerank_request_instance.to_dict()
# create an instance of RerankRequest from a dict
rerank_request_from_dict = RerankRequest.from_dict(rerank_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


