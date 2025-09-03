# RerankResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**results** | [**List[RerankResult]**](RerankResult.md) | Ranked results sorted by score (highest first) | [optional] 
**model** | **str** | Model used for reranking | [optional] 
**usage** | [**UsageInfo**](UsageInfo.md) |  | [optional] 
**object** | **str** | Response object type | [optional] [default to 'rerank']
**id** | **str** | Response ID | [optional] 
**created** | **int** | Creation timestamp | [optional] 

## Example

```python
from rerank_client_async.models.rerank_response import RerankResponse

# TODO update the JSON string below
json = "{}"
# create an instance of RerankResponse from a JSON string
rerank_response_instance = RerankResponse.from_json(json)
# print the JSON string representation of the object
print(RerankResponse.to_json())

# convert the object into a dict
rerank_response_dict = rerank_response_instance.to_dict()
# create an instance of RerankResponse from a dict
rerank_response_from_dict = RerankResponse.from_dict(rerank_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


