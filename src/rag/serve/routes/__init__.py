"""Route modules for the FastAPI service.

Each module exposes a single router that the lifespan's `create_app` mounts
under the right path. Keeping the route definitions here (rather than in
app.py) means each endpoint's request/response shape lives next to the
handler logic that fills it in.
"""
