"""POST /query - accept a question, return a generated answer.

Request body is `{"query": "..."}`. The `k` parameter is server-controlled
from the snapshot's retriever config; clients cannot override it. The
response is `{"query": "...", "answer": "..."}` - retrieval hits and
prompts are deliberately not exposed on the public endpoint, so a
leaked public key cannot be used to inspect the underlying chunks.
"""

from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, Request

from ..auth import require_role

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")


class QueryResponse(BaseModel):
    query: str
    answer: str


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_role("public"))],
)
def query(request: Request, body: QueryRequest) -> QueryResponse:
    pipeline = request.app.state.pipeline
    cfg = pipeline._snapshot.base_config
    k = cfg["retriever"].get("top_k", 3)

    result = pipeline.query(body.query, k=k)
    return QueryResponse(query=result["query"], answer=result["answer"])
