import json
import logging
import os

import azure.functions as func
from azure.core.credentials import AzureKeyCredential
from azure.functions.decorators.core import McpPropertyType
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

search_bp = func.Blueprint()


def _get_search_client() -> tuple[SearchClient, str, str]:
    service_name = os.getenv("AI_SEARCH_SERVICE_NAME")
    index_name = os.getenv("AI_SEARCH_INDEX_NAME", "earth-at-night")
    vector_field = os.getenv("AI_SEARCH_VECTOR_FIELD", "page_embedding_text_3_large")
    endpoint = f"https://{service_name}.search.windows.net" if service_name else None

    if not endpoint:
        raise ValueError(
            "AI_SEARCH_SERVICE_NAME is not configured. Set it in local.settings.json or the Azure Function app settings."
        )

    api_key = os.getenv("AI_SEARCH_API_KEY")
    credential = AzureKeyCredential(api_key) if api_key else DefaultAzureCredential()

    return (
        SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential,
        ),
        vector_field,
        index_name,
    )


@search_bp.mcp_tool()
@search_bp.mcp_tool_property(
    arg_name="query",
    description="Text to search for in the Azure AI Search index.",
    property_type=McpPropertyType.STRING,
    is_required=True,
)
@search_bp.mcp_tool_property(
    arg_name="top_k",
    description="Maximum number of matching chunks to return from the index.",
    property_type=McpPropertyType.INTEGER,
    is_required=False,
)
async def query_index(query: str, top_k: int | None = None) -> str:
    """Run a vector search in Azure AI Search and return the matching chunks."""
    index_name = os.getenv("AI_SEARCH_INDEX_NAME", "earth-at-night")

    try:
        top_k = top_k or int(os.getenv("AI_SEARCH_TOP_K", "5"))
        if top_k < 1:
            top_k = 1

        client, vector_field, index_name = _get_search_client()

        vector_query = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=top_k,
            fields=vector_field,
        )

        results = client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["id", "page_chunk", "page_number"],
            top=top_k,
            query_type="semantic",
        )

        matches = []
        for doc in results:
            matches.append(
                {
                    "id": doc.get("id"),
                    "page_number": doc.get("page_number"),
                    "page_chunk": doc.get("page_chunk"),
                    "score": doc.get("@search.score"),
                }
            )

        return json.dumps(
            {
                "query": query,
                "index": index_name,
                "results": matches,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logging.exception("AI Search query failed")
        return json.dumps(
            {
                "query": query,
                "index": index_name,
                "error": str(exc),
            },
            ensure_ascii=False,
        )
