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
