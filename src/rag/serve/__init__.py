"""FastAPI service package for Project_Ramsay.

The service wraps the existing Pipeline classes in an HTTP surface. The
lifespan loads the latest sealed snapshot, resolves its dataset through
the registry, eager-loads the generator and (for hybrid) the cross-encoder
reranker, and stores the warm Pipeline on `app.state.pipeline`. Endpoints
read that Pipeline and never reach for the registry themselves.
"""
