# rerank_client_sync.DefaultApi

All URIs are relative to *http://localhost:8080/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**rerank**](DefaultApi.md#rerank) | **POST** /rerank | Rerank documents against a query


# **rerank**
> RerankResponse rerank(rerank_request)

Rerank documents against a query

### Example


```python
import rerank_client_sync
from rerank_client_sync.models.rerank_request import RerankRequest
from rerank_client_sync.models.rerank_response import RerankResponse
from rerank_client_sync.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost:8080/v1
# See configuration.py for a list of all supported configuration parameters.
configuration = rerank_client_sync.Configuration(
    host = "http://localhost:8080/v1"
)


# Enter a context with an instance of the API client
with rerank_client_sync.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = rerank_client_sync.DefaultApi(api_client)
    rerank_request = rerank_client_sync.RerankRequest() # RerankRequest | 

    try:
        # Rerank documents against a query
        api_response = api_instance.rerank(rerank_request)
        print("The response of DefaultApi->rerank:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->rerank: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **rerank_request** | [**RerankRequest**](RerankRequest.md)|  | 

### Return type

[**RerankResponse**](RerankResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful rerank response |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

