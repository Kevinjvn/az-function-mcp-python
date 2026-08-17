from dataclasses import dataclass

import azure.functions as func


@dataclass
@func.mcp_content
class SearchMatch:
    """A single retrieved chunk from the AI Search index, with retrieval and reranking signals."""

    id: str
    """Unique identifier of the source document/chunk."""

    page_number: int
    """The page number in the source document this chunk was extracted from."""

    page_chunk: str
    """The verbatim text content of this chunk. Use this as the grounding source when answering."""

    score: float
    """The raw vector/BM25 retrieval score. Higher is more relevant, but not calibrated across queries — use rerankerScore for relative confidence."""

    rerankerScore: float
    """The semantic reranker score (0-4 scale typical for Azure AI Search semantic ranking). Prefer this over score when deciding how much to trust a result."""


@dataclass
@func.mcp_content
class SearchResponse:
    """Result of a semantic search against an Azure AI Search index."""

    query: str
    """The original search query text."""

    index: str
    """The name of the AI Search index that was queried."""

    results: list[SearchMatch]
    """Ranked list of matching chunks, best match first."""
