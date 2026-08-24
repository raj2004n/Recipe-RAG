"""Shared fixtures for the FastAPI service tests.

The tests do not run a real Pipeline. They patch `_build_pipeline` so the
lifespan stores a stub Pipeline on `app.state.pipeline`, and they set
`RAMSAY_API_KEYS` to a known map before the app is created. The
`client` fixture is the only thing test modules need to import directly.
"""
