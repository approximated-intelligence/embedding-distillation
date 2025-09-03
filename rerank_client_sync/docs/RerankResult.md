# RerankResult


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**score** | **float** | Relevance score for the document | [optional] 
**document** | **str** | The document text (if return_documents was true) | [optional] 
**index** | **int** | Original index of the document in the request | [optional] 
**meta_info** | **Dict[str, object]** | Additional metadata about the ranking | [optional] 

## Example

```python
from rerank_client_sync.models.rerank_result import RerankResult

# TODO update the JSON string below
json = "{}"
# create an instance of RerankResult from a JSON string
rerank_result_instance = RerankResult.from_json(json)
# print the JSON string representation of the object
print(RerankResult.to_json())

# convert the object into a dict
rerank_result_dict = rerank_result_instance.to_dict()
# create an instance of RerankResult from a dict
rerank_result_from_dict = RerankResult.from_dict(rerank_result_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


