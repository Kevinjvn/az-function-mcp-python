import json
import logging
import os

import azure.functions as func
from azure.core.credentials import AzureKeyCredential
from azure.functions.decorators.core import McpPropertyType
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from app.models import SearchMatch, SearchResponse

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


@search_bp.mcp_tool(use_result_schema=True)
@search_bp.mcp_tool_property(
    arg_name="query",
    description="Text to search for in the Azure AI Search index.",
    property_type=McpPropertyType.STRING,
    is_required=True,
)
async def query_index(query: str) -> SearchResponse:
    """Run a vector search in Azure AI Search and return the matching chunks."""
    index_name = os.getenv("AI_SEARCH_INDEX_NAME", "earth-at-night")
    min_reranker_score = float(os.getenv("AI_SEARCH_RERANKER_SCORE_THRESHOLD", "0.0"))
    min_retrieval_score = float(os.getenv("AI_SEARCH_RETRIEVAL_SCORE_THRESHOLD", "0.0"))

    try:
        client, vector_field, index_name = _get_search_client()

        vector_query = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=5,
            fields=vector_field,
        )

        results = client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["id", "page_chunk", "page_number"],
            top=5,
            query_type="semantic",
        )

        matches: list[SearchMatch] = []
        for doc in results:
            reranker_score = doc.get("@search.rerankerScore")
            retrieval_score = doc.get("@search.score")

            effective_score = (
                reranker_score if reranker_score is not None else retrieval_score
            )
            min_score = (
                min_reranker_score
                if reranker_score is not None
                else min_retrieval_score
            )

            if effective_score is not None and effective_score < min_score:
                continue

            matches.append(
                SearchMatch(
                    id=str(doc.get("id", "")),
                    page_number=int(doc.get("page_number", 0)),
                    page_chunk=str(doc.get("page_chunk", "")),
                    score=float(retrieval_score)
                    if retrieval_score is not None
                    else 0.0,
                    rerankerScore=float(reranker_score)
                    if reranker_score is not None
                    else 0.0,
                )
            )

        return SearchResponse(query=query, index=index_name, results=matches)
    except Exception as exc:
        logging.exception("AI Search query failed")
        return SearchResponse(query=query, index=index_name, results=[])
